// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-qfunction.hpp"

#include <sstream>
#include <string>

#include "ceed-occa-qfunctioncontext.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
QFunction::QFunction(const std::string &source, const std::string &function_name) : ceedIsIdentity(false) {
  filename      = source;
  qFunctionName = function_name;
}

QFunction *QFunction::getQFunction(CeedQFunction qf, const bool assertValid) {
  if (!qf) {
    return NULL;
  }

  QFunction *qFunction = NULL;

  CeedCallOcca(CeedQFunctionGetData(qf, &qFunction));

  return qFunction;
}

QFunction *QFunction::from(CeedQFunction qf) {
  QFunction *qFunction = getQFunction(qf);
  if (!qFunction) {
    return NULL;
  }

  CeedCallOcca(CeedQFunctionGetCeed(qf, &qFunction->ceed));

  CeedCallOcca(CeedQFunctionGetInnerContext(qf, &qFunction->qFunctionContext));

  CeedCallOcca(CeedQFunctionIsIdentity(qf, &qFunction->ceedIsIdentity));

  qFunction->args.setupQFunctionArgs(qf);
  if (!qFunction->args.isValid()) {
    return NULL;
  }

  return qFunction;
}

QFunction *QFunction::from(CeedOperator op) {
  if (!op) {
    return NULL;
  }

  CeedQFunction qf;

  CeedCallOcca(CeedOperatorGetQFunction(op, &qf));

  return QFunction::from(qf);
}

::occa::properties QFunction::getKernelProps(const CeedInt Q) {
  ::occa::properties props;

  // Types
  props["defines/CeedInt"]    = ::occa::dtype::get<CeedInt>().name();
  props["defines/CeedScalar"] = ::occa::dtype::get<CeedScalar>().name();

  // CEED defines
  props["defines/CeedPragmaSIMD"]     = "";
  props["defines/CEED_Q_VLA"]         = "OCCA_Q";
  props["defines/CEED_ERROR_SUCCESS"] = 0;

  std::stringstream ss;
  ss << "#define CEED_QFUNCTION(FUNC_NAME) \\" << std::endl
     << "  inline int FUNC_NAME" << std::endl
     << "#define CEED_QFUNCTION_HELPER \\" << std::endl
     << "  inline" << std::endl
     << std::endl
     << "#include \"" << filename << "\"" << std::endl;

  props["headers"].asArray() += ss.str();

  return props;
}

int QFunction::buildKernel(const CeedInt Q) {
  // TODO: Store a kernel per Q
  if (!qFunctionKernel.isInitialized()) {
    ::occa::properties props = getKernelProps(Q);

    // Properties only used in the QFunction kernel source
    props["defines/OCCA_Q"] = Q;

    const std::string kernelName = "qf_" + qFunctionName;

    qFunctionKernel = (getDevice().buildKernelFromString(getKernelSource(kernelName, Q), kernelName, props));
  }

  return CEED_ERROR_SUCCESS;
}

std::string QFunction::getKernelSource(const std::string &kernelName, const CeedInt Q) {
  std::stringstream ss;

  ss << "@kernel" << std::endl << "void " << kernelName << "(" << std::endl;

  // qfunction arguments
  for (int i = 0; i < args.inputCount(); ++i) {
    ss << "  const CeedScalar *in" << i << ',' << std::endl;
  }
  for (int i = 0; i < args.outputCount(); ++i) {
    ss << "  CeedScalar *out" << i << ',' << std::endl;
  }
  ss << "  void *ctx" << std::endl;
  ss << ") {" << std::endl;

  // Iterate over Q and call qfunction
  ss << "  @tile(128, @outer, @inner)" << std::endl
     << "  for (int q = 0; q < OCCA_Q; ++q) {" << std::endl
     << "    const CeedScalar* in[" << std::max(1, args.inputCount()) << "];" << std::endl
     << "    CeedScalar* out[" << std::max(1, args.outputCount()) << "];" << std::endl;

  // Set and define in for the q point
  for (int i = 0; i < args.inputCount(); ++i) {
    const CeedInt     fieldSize = args.getQfInput(i).size;
    const std::string qIn_i     = "qIn" + std::to_string(i);
    const std::string in_i      = "in" + std::to_string(i);

    ss << "    CeedScalar " << qIn_i << "[" << fieldSize << "];" << std::endl
       << "    in[" << i << "] = " << qIn_i << ";"
       << std::endl
       // Copy q data
       << "    for (int qi = 0; qi < " << fieldSize << "; ++qi) {" << std::endl
       << "      " << qIn_i << "[qi] = " << in_i << "[q + (OCCA_Q * qi)];" << std::endl
       << "    }" << std::endl;
  }

  // Set out for the q point
  for (int i = 0; i < args.outputCount(); ++i) {
    const CeedInt     fieldSize = args.getQfOutput(i).size;
    const std::string qOut_i    = "qOut" + std::to_string(i);

    ss << "    CeedScalar " << qOut_i << "[" << fieldSize << "];" << std::endl << "    out[" << i << "] = " << qOut_i << ";" << std::endl;
  }

  ss << "    " << qFunctionName << "(ctx, 1, in, out);" << std::endl;

  // Copy out for the q point
  for (int i = 0; i < args.outputCount(); ++i) {
    const CeedInt     fieldSize = args.getQfOutput(i).size;
    const std::string qOut_i    = "qOut" + std::to_string(i);
    const std::string out_i     = "out" + std::to_string(i);

    ss << "    for (int qi = 0; qi < " << fieldSize << "; ++qi) {" << std::endl
       << "      " << out_i << "[q + (OCCA_Q * qi)] = " << qOut_i << "[qi];" << std::endl
       << "    }" << std::endl;
  }

  ss << "  }" << std::endl << "}";

  return ss.str();
}

int QFunction::apply(CeedInt Q, CeedVector *U, CeedVector *V) {
  CeedCallBackend(buildKernel(Q));

  std::vector<CeedScalar *> outputArgs;

  qFunctionKernel.clearArgs();

  for (CeedInt i = 0; i < args.inputCount(); i++) {
    Vector *u = Vector::from(U[i]);
    if (!u) {
      return ceedError("Incorrect qFunction input field: U[" + std::to_string(i) + "]");
    }
    qFunctionKernel.pushArg(u->getConstKernelArg());
  }

  for (CeedInt i = 0; i < args.outputCount(); i++) {
    Vector *v = Vector::from(V[i]);
    if (!v) {
      return ceedError("Incorrect qFunction output field: V[" + std::to_string(i) + "]");
    }
    qFunctionKernel.pushArg(v->getKernelArg());
  }
  if (qFunctionContext) {
    QFunctionContext *ctx = QFunctionContext::from(qFunctionContext);
    qFunctionKernel.pushArg(ctx->getKernelArg());
  } else {
    qFunctionKernel.pushArg(::occa::null);
  }

  qFunctionKernel.run();

  return CEED_ERROR_SUCCESS;
}

//---[ Ceed Callbacks ]-----------
int QFunction::registerCeedFunction(Ceed ceed, CeedQFunction qf, const char *fname, ceed::occa::ceedFunction f) {
  return CeedSetBackendFunction(ceed, "QFunction", qf, fname, f);
}

int QFunction::ceedCreate(CeedQFunction qf) {
  Ceed ceed;
  CeedCallBackend(CeedQFunctionGetCeed(qf, &ceed));
  Context *context;
  CeedCallBackend(CeedGetData(ceed, &context));
  char *source;
  CeedCallBackend(CeedQFunctionGetSourcePath(qf, &source));
  char *function_name;
  CeedCallBackend(CeedQFunctionGetKernelName(qf, &function_name));

  QFunction *qFunction = new QFunction(source, function_name);
  CeedCallBackend(CeedQFunctionSetData(qf, qFunction));

  CeedOccaRegisterFunction(qf, "Apply", QFunction::ceedApply);
  CeedOccaRegisterFunction(qf, "Destroy", QFunction::ceedDestroy);

  return CEED_ERROR_SUCCESS;
}

int QFunction::ceedApply(CeedQFunction qf, CeedInt Q, CeedVector *U, CeedVector *V) {
  QFunction *qFunction = QFunction::from(qf);
  if (qFunction) {
    return qFunction->apply(Q, U, V);
  }

  return 1;
}

int QFunction::ceedDestroy(CeedQFunction qf) {
  delete getQFunction(qf, false);
  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed
