// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include "ceed-occa-cpu-operator.hpp"

#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-qfunction.hpp"
#include "ceed-occa-qfunctioncontext.hpp"
#include "ceed-occa-simplex-basis.hpp"
#include "ceed-occa-tensor-basis.hpp"

#define CEED_OCCA_PRINT_KERNEL_HASHES 0

namespace ceed {
namespace occa {
CpuOperator::CpuOperator() {}

CpuOperator::~CpuOperator() {}

void CpuOperator::setupVectors() {
  setupVectors(args.inputCount(), args.opInputs, args.qfInputs, dofInputs);
  setupVectors(args.outputCount(), args.opOutputs, args.qfOutputs, dofOutputs);
}

void CpuOperator::setupVectors(const int fieldCount, OperatorFieldVector &opFields, QFunctionFieldVector &qfFields, VectorVector &vectors) {
  for (int i = 0; i < fieldCount; ++i) {
    const QFunctionField &qfField = qfFields[i];
    const OperatorField  &opField = opFields[i];

    if (qfField.evalMode == CEED_EVAL_WEIGHT) {
      // Weight kernel doesn't use the input
      vectors.push_back(NULL);
      continue;
    }

    int entries;
    if (qfField.evalMode == CEED_EVAL_NONE) {
      // The output vector stores values at quadrature points
      entries = (ceedElementCount * ceedQ * qfField.size);
    } else {
      // The output vector stores the element dof values
      entries = (ceedElementCount * opField.getElementSize() * opField.getComponentCount());
    }

    Vector *dofVector = new Vector();
    dofVector->ceed   = ceed;
    dofVector->resize(entries);

    vectors.push_back(dofVector);
  }
}

void CpuOperator::freeVectors() {
  for (int i = 0; i < args.inputCount(); ++i) {
    delete dofInputs[i];
  }
  for (int i = 0; i < args.outputCount(); ++i) {
    delete dofOutputs[i];
  }
  dofInputs.clear();
  dofOutputs.clear();
}

void CpuOperator::setupInputs(Vector *in) {
  for (int i = 0; i < args.inputCount(); ++i) {
    // Weight kernel doesn't use the input vector
    if (args.getInputEvalMode(i) == CEED_EVAL_WEIGHT) {
      continue;
    }

    const OperatorField &opField = args.getOpInput(i);

    Vector *input  = opField.usesActiveVector() ? in : opField.vec;
    Vector *output = dofInputs[i];

    opField.elemRestriction->apply(CEED_NOTRANSPOSE, *input, *output);
  }
}

void CpuOperator::setupOutputs(Vector *out) {
  for (int i = 0; i < args.outputCount(); ++i) {
    // Weight is not supported for output vectors
    if (args.getOutputEvalMode(i) == CEED_EVAL_WEIGHT) {
      continue;
    }

    const OperatorField &opField = args.getOpOutput(i);

    Vector *input  = dofOutputs[i];
    Vector *output = opField.usesActiveVector() ? out : opField.vec;

    opField.elemRestriction->apply(CEED_TRANSPOSE, *input, *output);
  }
}

void CpuOperator::applyQFunction() {
  if (qfunction->qFunctionContext) {
    QFunctionContext *ctx = QFunctionContext::from(qfunction->qFunctionContext);
    applyAddKernel.pushArg(ctx->getKernelArg());
  } else {
    applyAddKernel.pushArg(::occa::null);
  }
  applyAddKernel.pushArg(ceedElementCount);

  for (int i = 0; i < args.inputCount(); ++i) {
    const bool isInput = true;
    pushKernelArgs(dofInputs[i], isInput, i);
  }

  for (int i = 0; i < args.outputCount(); ++i) {
    const bool isInput = false;
    pushKernelArgs(dofOutputs[i], isInput, i);
  }

  applyAddKernel.run();
}

void CpuOperator::pushKernelArgs(Vector *vec, const bool isInput, const int index) {
  const OperatorField  &opField = args.getOpField(isInput, index);
  const QFunctionField &qfField = args.getQfField(isInput, index);

  if (opField.hasBasis()) {
    if (opField.usingTensorBasis()) {
      pushTensorBasisKernelArgs(qfField, *((TensorBasis *)opField.basis));
    } else {
      pushSimplexBasisKernelArgs(qfField, *((SimplexBasis *)opField.basis));
    }
  }

  if (vec) {
    if (isInput) {
      applyAddKernel.pushArg(vec->getConstKernelArg());
    } else {
      applyAddKernel.pushArg(vec->getKernelArg());
    }
  } else {
    applyAddKernel.pushArg(::occa::null);
  }
}

void CpuOperator::pushTensorBasisKernelArgs(const QFunctionField &qfField, TensorBasis &basis) {
  switch (qfField.evalMode) {
    case CEED_EVAL_INTERP: {
      applyAddKernel.pushArg(basis.interp1D);
      break;
    }
    case CEED_EVAL_GRAD: {
      applyAddKernel.pushArg(basis.interp1D);
      applyAddKernel.pushArg(basis.grad1D);
      break;
    }
    case CEED_EVAL_WEIGHT: {
      applyAddKernel.pushArg(basis.qWeight1D);
      break;
    }
    default: {
    }
  }
}

void CpuOperator::pushSimplexBasisKernelArgs(const QFunctionField &qfField, SimplexBasis &basis) {
  switch (qfField.evalMode) {
    case CEED_EVAL_INTERP: {
      applyAddKernel.pushArg(basis.interp);
      break;
    }
    case CEED_EVAL_GRAD: {
      applyAddKernel.pushArg(basis.grad);
      break;
    }
    case CEED_EVAL_WEIGHT: {
      applyAddKernel.pushArg(basis.qWeight);
      break;
    }
    default: {
    }
  }
}

::occa::properties CpuOperator::getKernelProps() {
  ::occa::properties props = qfunction->getKernelProps(ceedQ);

  props["defines/OCCA_Q"] = ceedQ;

  return props;
}

void CpuOperator::applyAdd(Vector *in, Vector *out) {
  // Setup helper vectors
  setupVectors();

  // Dof nodes -> local dofs
  setupInputs(in);

  // Apply qFunction
  applyQFunction();

  // Local dofs -> dof nodes
  setupOutputs(out);

  // Cleanup helper vectors
  freeVectors();
}

::occa::kernel CpuOperator::buildApplyAddKernel() {
  std::stringstream ss;

  addBasisFunctionSource(ss);

  addKernelSource(ss);

  const std::string kernelSource = ss.str();

  CeedDebug(ceed, kernelSource.c_str());

  // TODO: Store a kernel per Q
  return getDevice().buildKernelFromString(kernelSource, "applyAdd", getKernelProps());
}

//---[ Kernel Generation ]--------------------
void CpuOperator::addBasisFunctionSource(std::stringstream &ss) {
  BasisVector sourceBasis;
  for (int i = 0; i < args.inputCount(); ++i) {
    addBasisIfMissingSource(sourceBasis, args.getOpInput(i).basis);
  }
  for (int i = 0; i < args.outputCount(); ++i) {
    addBasisIfMissingSource(sourceBasis, args.getOpOutput(i).basis);
  }

  // Make sure there's a break between past code
  ss << std::endl;

  // Add source code for each unique basis function
  const int basisCount = (int)sourceBasis.size();
  for (int i = 0; i < basisCount; ++i) {
    Basis &basis = *(sourceBasis[i]);

    ss << "// Code generation for basis " << i + 1 << std::endl << "//---[ START ]-------------------------------" << std::endl;

    // Undefine and redefine required variables
    if (basis.isTensorBasis()) {
      TensorBasis &basisTensor = (TensorBasis &)basis;
      ss << "#undef  TENSOR_FUNCTION" << std::endl
         << "#undef  P1D" << std::endl
         << "#undef  Q1D" << std::endl
         << "#define P1D " << basisTensor.P1D << std::endl
         << "#define Q1D " << basisTensor.Q1D << std::endl;
    } else {
      SimplexBasis &basisSimplex = (SimplexBasis &)basis;
      ss << "#undef  SIMPLEX_FUNCTION" << std::endl
         << "#undef  DIM" << std::endl
         << "#undef  P" << std::endl
         << "#undef  Q" << std::endl
         << "#define DIM " << basisSimplex.dim << std::endl
         << "#define P   " << basisSimplex.P << std::endl
         << "#define Q   " << basisSimplex.Q << std::endl;
    }

    ss << std::endl << basis.getFunctionSource() << std::endl << "//---[ END ]---------------------------------" << std::endl;
  }
}

void CpuOperator::addBasisIfMissingSource(BasisVector &sourceBasis, Basis *basis) {
  // Avoid adding duplicate sources which will result in colliding symbol names

  // No basis
  if (!basis) {
    return;
  }

  // Fast enough since we expect a small number of inputs/outputs
  const int existingBasisCount = (int)sourceBasis.size();
  for (int i = 0; i < existingBasisCount; ++i) {
    Basis *other = sourceBasis[i];
    // They are different basis types so other != basis
    if (basis->isTensorBasis() != other->isTensorBasis()) {
      continue;
    }

    if (basis->dim == other->dim && basis->P == other->P && basis->Q == other->Q) {
      // `other` wil generate the same code
      return;
    }
  }

  // Basis didn't match any other existing basis
  sourceBasis.push_back(basis);
}

void CpuOperator::addKernelSource(std::stringstream &ss) {
  // Make sure there's a break between past code
  ss << std::endl;

  ss << "@kernel void applyAdd(" << std::endl;

  addKernelArgsSource(ss);

  ss << std::endl
     << ") {" << std::endl
     << "  @tile(128, @outer, @inner)" << std::endl
     << "  for (int element = 0; element < elementCount; ++element) {" << std::endl;

#if CEED_OCCA_PRINT_KERNEL_HASHES
  // Print to see which kernel is being run
  ss << "    if (element == 0) {" << std::endl
     << "      printf(\"\\n\\nOperator Kernel: \" OKL_KERNEL_HASH \"\\n\\n\");" << std::endl
     << "    }" << std::endl;
#endif

  addQuadArraySource(ss);

  ss << std::endl << "    // [Start] Transforming inputs to quadrature points" << std::endl;
  addInputSetupSource(ss);
  ss << "    // [End] Transforming inputs to quadrature points" << std::endl << std::endl;

  addQFunctionApplicationSource(ss);

  ss << std::endl << "    // [Start] Transforming outputs to quadrature points" << std::endl;
  addOutputSetupSource(ss);
  ss << "    // [End] Transforming outputs to quadrature points" << std::endl;

  ss << "  }" << std::endl << "}" << std::endl;
}

void CpuOperator::addKernelArgsSource(std::stringstream &ss) {
  ss << "  void *ctx," << std::endl << "  const CeedInt elementCount";

  for (int i = 0; i < args.inputCount(); ++i) {
    const bool isInput = true;
    addKernelArgSource(ss, isInput, i);
  }
  for (int i = 0; i < args.outputCount(); ++i) {
    const bool isInput = false;
    addKernelArgSource(ss, isInput, i);
  }
}

void CpuOperator::addKernelArgSource(std::stringstream &ss, const bool isInput, const int index) {
  const OperatorField  &opField = args.getOpField(isInput, index);
  const QFunctionField &qfField = args.getQfField(isInput, index);

  std::stringstream dimAttribute;
  if (opField.hasBasis()) {
    ss << ',' << std::endl;
    if (opField.usingTensorBasis()) {
      addTensorKernelArgSource(ss, isInput, index, opField, qfField, dimAttribute);
    } else {
      addSimplexKernelArgSource(ss, isInput, index, opField, qfField, dimAttribute);
    }
  }

  ss << ',' << std::endl;
  if (isInput) {
    ss << "  const CeedScalar *" << dofInputVar(index) << dimAttribute.str();
  } else {
    ss << "  CeedScalar *" << dofOutputVar(index) << dimAttribute.str();
  }
}

void CpuOperator::addTensorKernelArgSource(std::stringstream &ss, const bool isInput, const int index, const OperatorField &opField,
                                           const QFunctionField &qfField, std::stringstream &dimAttribute) {
  TensorBasis &basis = *((TensorBasis *)opField.basis);

  dimAttribute << " @dim(";

  if (qfField.evalMode == CEED_EVAL_INTERP) {
    ss << "  const CeedScalar *" << interpVar(isInput, index);

    // @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount)
    for (int i = 0; i < basis.dim; ++i) {
      dimAttribute << basis.P1D << ", ";
    }
    dimAttribute << basis.ceedComponentCount << ", elementCount";
  } else if (qfField.evalMode == CEED_EVAL_GRAD) {
    ss << "  const CeedScalar *" << interpVar(isInput, index) << ',' << std::endl << "  const CeedScalar *" << gradVar(isInput, index);

    // @dim(P1D, P1D, BASIS_COMPONENT_COUNT, elementCount)
    for (int i = 0; i < basis.dim; ++i) {
      dimAttribute << basis.P1D << ", ";
    }
    dimAttribute << basis.ceedComponentCount << ", elementCount";
  } else if (qfField.evalMode == CEED_EVAL_WEIGHT) {
    ss << "  const CeedScalar *" << qWeightVar(isInput, index);

    // @dim(Q1D, Q1D, elementCount)
    for (int i = 0; i < basis.dim; ++i) {
      dimAttribute << basis.Q1D << ", ";
    }
    dimAttribute << "elementCount";
  } else {
    // Clear @dim
    dimAttribute.str("");
    return;
  }

  dimAttribute << ")";
}

void CpuOperator::addSimplexKernelArgSource(std::stringstream &ss, const bool isInput, const int index, const OperatorField &opField,
                                            const QFunctionField &qfField, std::stringstream &dimAttribute) {
  SimplexBasis &basis = *((SimplexBasis *)opField.basis);

  dimAttribute << " @dim(";

  if (qfField.evalMode == CEED_EVAL_INTERP) {
    ss << "  const CeedScalar *" << interpVar(isInput, index);

    // @dim(P, BASIS_COMPONENT_COUNT, elementCount)
    dimAttribute << basis.P << ", " << basis.ceedComponentCount << ", elementCount";
  } else if (qfField.evalMode == CEED_EVAL_GRAD) {
    ss << "  const CeedScalar *" << gradVar(isInput, index);

    // @dim(P, BASIS_COMPONENT_COUNT, elementCount)
    dimAttribute << basis.P << ", " << basis.ceedComponentCount << ", elementCount";
  } else if (qfField.evalMode == CEED_EVAL_WEIGHT) {
    ss << "  const CeedScalar *" << qWeightVar(isInput, index);

    // @dim(Q, elementCount)
    dimAttribute << basis.Q << ", "
                 << "elementCount";
  } else {
    // Clear @dim
    dimAttribute.str("");
    return;
  }

  dimAttribute << ")";
}

void CpuOperator::addQuadArraySource(std::stringstream &ss) {
  const int inputs  = args.inputCount();
  const int outputs = args.outputCount();

  const std::string quadInput  = "quadInput";
  const std::string quadOutput = "quadOutput";

  ss << "    // Store the transformed input quad values" << std::endl;
  for (int i = 0; i < inputs; ++i) {
    const bool isInput = true;
    addSingleQfunctionQuadArraySource(ss, isInput, i, quadInput);
  }

  ss << std::endl << "    // Store the transformed output quad values" << std::endl;
  for (int i = 0; i < outputs; ++i) {
    const bool isInput = false;
    addSingleQfunctionQuadArraySource(ss, isInput, i, quadOutput);
  }
  ss << std::endl;

  ss << std::endl << "    // Store all input pointers in a single array" << std::endl;
  addQfunctionQuadArraySource(ss, true, inputs, quadInput);

  ss << std::endl << "    // Store all output pointers in a single array" << std::endl;
  addQfunctionQuadArraySource(ss, false, outputs, quadOutput);

  ss << std::endl;
}

void CpuOperator::addSingleQfunctionQuadArraySource(std::stringstream &ss, const bool isInput, const int index, const std::string &name) {
  // Output:
  //   CeedScalar quadInput0[DIM][COMPONENTS][OCCA_Q];
  //   CeedScalar quadInput0[OCCA_Q * SIZE];

  const OperatorField &opField  = args.getOpField(isInput, index);
  CeedEvalMode         evalMode = args.getEvalMode(isInput, index);

  if (evalMode == CEED_EVAL_GRAD) {
    ss << "    CeedScalar " << indexedVar(name, index) << "[" << opField.getDim() << "]"
       << "[" << opField.getComponentCount() << "]"
       << "[OCCA_Q];" << std::endl;
  } else if (evalMode == CEED_EVAL_INTERP) {
    ss << "    CeedScalar " << indexedVar(name, index) << "[" << opField.getComponentCount() << "]"
       << "[OCCA_Q];" << std::endl;
  } else {
    const QFunctionField &qfField = args.getQfField(isInput, index);

    ss << "    CeedScalar " << indexedVar(name, index) << "[OCCA_Q * " << qfField.size << "];" << std::endl;
  }
}

void CpuOperator::addQfunctionQuadArraySource(std::stringstream &ss, const bool isInput, const int count, const std::string &name) {
  // Output:
  //   CeedScalar *quadInputs[2] = {
  //     (CeedScalar*) quadInput0,
  //     (CeedScalar*) quadInput1
  //   };

  // Add an 's': quadInput -> quadInputs
  const std::string arrayName = name + "s";

  ss << "    CeedScalar *" << arrayName << "[" << count << "] = {" << std::endl;
  for (int i = 0; i < count; ++i) {
    if (i) {
      ss << ',' << std::endl;
    }
    ss << "      (CeedScalar*) " << indexedVar(name, i);
  }
  ss << std::endl << "    };" << std::endl;
}

void CpuOperator::addInputSetupSource(std::stringstream &ss) {
  const bool isInput = true;
  addBasisApplySource(ss, isInput, args.inputCount());
}

void CpuOperator::addOutputSetupSource(std::stringstream &ss) {
  const bool isInput = false;
  addBasisApplySource(ss, isInput, args.outputCount());
}

void CpuOperator::addBasisApplySource(std::stringstream &ss, const bool isInput, const int count) {
  for (int i = 0; i < count; ++i) {
    CeedEvalMode evalMode = args.getEvalMode(isInput, i);

    if (evalMode == CEED_EVAL_INTERP) {
      addInterpSource(ss, isInput, i);
    } else if (evalMode == CEED_EVAL_GRAD) {
      const bool hasTensorBasis = args.getOpField(isInput, i).usingTensorBasis();
      if (hasTensorBasis) {
        addGradTensorSource(ss, isInput, i);
      } else {
        addGradSimplexSource(ss, isInput, i);
      }
    } else if (evalMode == CEED_EVAL_WEIGHT) {
      addWeightSource(ss, isInput, i);
    } else if (evalMode == CEED_EVAL_NONE) {
      addCopySource(ss, isInput, i);
    }
  }
}

void CpuOperator::addInterpSource(std::stringstream &ss, const bool isInput, const int index) {
  const OperatorField &opField          = args.getOpField(isInput, index);
  const bool           usingTensorBasis = opField.usingTensorBasis();
  const int            components       = opField.getComponentCount();
  const int            dim              = opField.getDim();

  const std::string weights = interpVar(isInput, index);

  std::string dimArgs;
  if (usingTensorBasis) {
    for (int i = 0; i < dim; ++i) {
      if (i) {
        dimArgs += ", ";
      }
      dimArgs += '0';
    }
  } else {
    dimArgs = "0";
  }

  std::string input, output;
  if (isInput) {
    input  = "&" + dofInputVar(index) + "(" + dimArgs + ", component, element)";
    output = "(CeedScalar*) " + indexedVar("quadInput", index) + "[component]";
  } else {
    input  = "(CeedScalar*) " + indexedVar("quadOutput", index) + "[component]";
    output = "&" + dofOutputVar(index) + "(" + dimArgs + ", component, element)";
  }

  ss << "    // Applying interp (" << xputName(isInput) << ": " << index << ")" << std::endl
     << "    for (int component = 0; component < " << components << "; ++component) {" << std::endl
     << "      " << elementFunction(isInput, index) << "(" << std::endl
     << "        " << weights << ',' << std::endl
     << "        " << input << ',' << std::endl
     << "        " << output << std::endl
     << "      );" << std::endl
     << "    }" << std::endl
     << std::endl;
}

void CpuOperator::addGradTensorSource(std::stringstream &ss, const bool isInput, const int index) {
  const OperatorField &opField    = args.getOpField(isInput, index);
  const int            components = opField.getComponentCount();
  const int            dim        = opField.getDim();

  const std::string B  = interpVar(isInput, index);
  const std::string Bx = gradVar(isInput, index);

  std::string dimArgs;
  for (int i = 0; i < dim; ++i) {
    if (i) {
      dimArgs += ", ";
    }
    dimArgs += '0';
  }

  std::string inputs, outputs;
  if (isInput) {
    inputs = "&" + dofInputVar(index) + "(" + dimArgs + ", component, element)";

    for (int i = 0; i < dim; ++i) {
      if (i) {
        outputs += ",\n        ";
      }
      const std::string iStr = std::to_string(i);
      outputs += "(CeedScalar*) " + indexedVar("quadInput", index) + "[" + iStr + "][component]";
    }
  } else {
    for (int i = 0; i < dim; ++i) {
      if (i) {
        inputs += ",\n        ";
      }
      const std::string iStr = std::to_string(i);
      inputs += "(CeedScalar*) " + indexedVar("quadOutput", index) + "[" + iStr + "][component]";
    }

    outputs = "&" + dofOutputVar(index) + "(" + dimArgs + ", component, element)";
  }

  ss << "    // Applying grad-tensor (" << xputName(isInput) << ": " << index << ")" << std::endl
     << "    for (int component = 0; component < " << components << "; ++component) {" << std::endl
     << "      " << elementFunction(isInput, index) << "(" << std::endl
     << "        " << B << ',' << std::endl
     << "        " << Bx << ',' << std::endl
     << "        " << inputs << ',' << std::endl
     << "        " << outputs << std::endl
     << "      );" << std::endl
     << "    }" << std::endl
     << std::endl;
}

void CpuOperator::addGradSimplexSource(std::stringstream &ss, const bool isInput, const int index) {
  const int components = (args.getOpField(isInput, index).getComponentCount());

  const std::string weights = gradVar(isInput, index);

  std::string input, output;
  if (isInput) {
    input  = "&" + dofInputVar(index) + "(0, component, element)";
    output = "(CeedScalar*) " + indexedVar("quadInput", index) + "[component]";
  } else {
    input  = "(CeedScalar*) " + indexedVar("quadOutput", index) + "[component]";
    output = "&" + dofOutputVar(index) + "(0, component, element)";
  }

  ss << "    // Applying grad-simplex (" << xputName(isInput) << ": " << index << ")" << std::endl
     << "    for (int component = 0; component < " << components << "; ++component) {" << std::endl
     << "      " << elementFunction(isInput, index) << "(" << std::endl
     << "        " << weights << ',' << std::endl
     << "        " << input << ',' << std::endl
     << "        " << output << std::endl
     << "      );" << std::endl
     << "    }" << std::endl
     << std::endl;
}

void CpuOperator::addWeightSource(std::stringstream &ss, const bool isInput, const int index) {
  const std::string weights = qWeightVar(isInput, index);

  std::string output;
  if (isInput) {
    // TODO: Can the weight operator handle multiple components?
    output = "(CeedScalar*) " + indexedVar("quadInput", index);
  } else {
    output = "&" + dofOutputVar(index) + "(0, element)";
  }

  ss << "    // Applying weight (" << xputName(isInput) << ": " << index << ")" << std::endl
     << "    " << elementFunction(isInput, index) << "(" << std::endl
     << "      " << weights << ',' << std::endl
     << "      " << output << std::endl
     << "    );" << std::endl
     << std::endl;
}

void CpuOperator::addCopySource(std::stringstream &ss, const bool isInput, const int index) {
  const QFunctionField &qfField = args.getQfField(isInput, index);
  const std::string     size    = std::to_string(qfField.size);

  std::string input, output;
  if (isInput) {
    input += dofInputVar(index) + "[q + (OCCA_Q * (field + element * " + size + "))]";
    output += indexedVar("quadInput", index) + "[q + field * OCCA_Q]";
  } else {
    input  = indexedVar("quadOutput", index) + "[q + field * OCCA_Q]";
    output = dofOutputVar(index) + "[q + (OCCA_Q * (field + element * " + size + "))]";
  }

  ss << "    // Copying source directly (" << xputName(isInput) << ": " << index << ")" << std::endl
     << "    for (int field = 0; field < " << size << "; ++field) {" << std::endl
     << "      for (int q = 0; q < OCCA_Q; ++q) {" << std::endl
     << "        " << output << " = " << input << ";" << std::endl
     << "      }" << std::endl
     << "    }" << std::endl
     << std::endl;
}

void CpuOperator::addQFunctionApplicationSource(std::stringstream &ss) {
  ss << "    // Apply qFunction" << std::endl
     << "    " << qfunction->qFunctionName << "(ctx, OCCA_Q, quadInputs, quadOutputs);" << std::endl
     << std::endl;
}

//  ---[ Variables ]-----------------
std::string CpuOperator::elementFunction(const bool isInput, const int index) {
  return fullFieldFunctionName(isInput, args.getOpField(isInput, index), args.getQfField(isInput, index));
}

std::string CpuOperator::fieldFunctionName(const QFunctionField &qfField) {
  switch (qfField.evalMode) {
    case CEED_EVAL_INTERP:
      return "interp";
    case CEED_EVAL_GRAD:
      return "grad";
    case CEED_EVAL_WEIGHT:
      return "weight";
    default:
      return "none";
  }
}

std::string CpuOperator::fullFieldFunctionName(const bool isInput, const OperatorField &opField, const QFunctionField &qfField) {
  // Output:
  //   - tensor_1d_interpElement_Q2_P2
  //   - simplex_1d_interpElementTranspose_Q2_P2

  const bool        usingTensorBasis = opField.usingTensorBasis();
  std::stringstream ss;
  int               dim, Q, P;

  if (usingTensorBasis) {
    TensorBasis &basis = *((TensorBasis *)opField.basis);
    dim                = basis.dim;
    Q                  = basis.Q1D;
    P                  = basis.P1D;
    ss << "tensor_";
  } else {
    SimplexBasis &basis = *((SimplexBasis *)opField.basis);
    dim                 = basis.dim;
    Q                   = basis.Q;
    P                   = basis.P;
    ss << "simplex_";
  }

  ss << dim << "d_" << fieldFunctionName(qfField) << "Element";

  if (!isInput) {
    ss << "Transpose";
  }

  ss << "_Q" << Q << "_P" << P;

  return ss.str();
}
}  // namespace occa
}  // namespace ceed
