// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#warning "libCEED OCCA backend is experimental; for best performance, use device native backends"

#include <map>
#include <occa.hpp>
#include <vector>

#include "ceed-occa-context.hpp"
#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-operator.hpp"
#include "ceed-occa-qfunction.hpp"
#include "ceed-occa-qfunctioncontext.hpp"
#include "ceed-occa-simplex-basis.hpp"
#include "ceed-occa-tensor-basis.hpp"
#include "ceed-occa-types.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
typedef std::map<std::string, std::string> StringMap;
typedef std::vector<std::string>           StringVector;

enum ResourceParserStep { RESOURCE, QUERY_KEY, QUERY_VALUE };

static const char RESOURCE_DELIMITER        = '/';
static const char QUERY_DELIMITER           = ':';
static const char QUERY_KEY_VALUE_DELIMITER = '=';
static const char QUERY_ARG_DELIMITER       = ',';

static std::string getDefaultDeviceMode(const bool cpuMode, const bool gpuMode) {
  // In case both cpuMode and gpuMode are set, prioritize the GPU if available
  // For example, if the resource is "/*/occa"
  if (gpuMode) {
    if (::occa::modeIsEnabled("CUDA")) {
      return "CUDA";
    }
    if (::occa::modeIsEnabled("HIP")) {
      return "HIP";
    }
    if (::occa::modeIsEnabled("dpcpp")) {
      return "dpcpp";
    }
    if (::occa::modeIsEnabled("OpenCL")) {
      return "OpenCL";
    }
    // Metal doesn't support doubles
  }

  if (cpuMode) {
    if (::occa::modeIsEnabled("OpenMP")) {
      return "OpenMP";
    }
    return "Serial";
  }

  return "";
}

static int getDeviceMode(const std::string &match, std::string &mode) {
  if (match == "cuda") {
    mode = "CUDA";
    return CEED_ERROR_SUCCESS;
  }
  if (match == "hip") {
    mode = "HIP";
    return CEED_ERROR_SUCCESS;
  }
  if (match == "dpcpp") {
    mode = "dpcpp";
    return CEED_ERROR_SUCCESS;
  }
  if (match == "opencl") {
    mode = "OpenCL";
    return CEED_ERROR_SUCCESS;
  }
  if (match == "openmp") {
    mode = "OpenMP";
    return CEED_ERROR_SUCCESS;
  }
  if (match == "serial") {
    mode = "Serial";
    return CEED_ERROR_SUCCESS;
  }

  const bool autoMode = match == "*";
  const bool cpuMode  = match == "cpu";
  const bool gpuMode  = match == "gpu";

  mode = getDefaultDeviceMode(cpuMode || autoMode, gpuMode || autoMode);
  return !mode.size();
}

static int splitCeedResource(const std::string &resource, std::string &match, StringMap &query) {
  /*
   * resource:
   *
   *    "/gpu/occa?mode='CUDA':device_id=0"
   *
   * resourceVector:
   *
   *    ["gpu", "occa"]
   *
   * match:
   *
   *    "gpu"
   *
   * query:
   *
   *    {
   *      "mode": "'CUDA'",
   *      "device_id": "0",
   *    }
   */
  const int   charCount  = (int)resource.size();
  const char *c_resource = resource.c_str();

  StringVector resourceVector;

  ResourceParserStep parsingStep = RESOURCE;
  int                wordStart   = 1;
  std::string        queryKey;

  // Check for /gpu/cuda/occa, /gpu/hip/occa, /cpu/self/occa, /cpu/openmp/occa
  // Note: added for matching style with other backends
  if (resource == "/gpu/cuda/occa") {
    match = "cuda";
    return CEED_ERROR_SUCCESS;
  }
  if (resource == "/gpu/hip/occa") {
    match = "hip";
    return CEED_ERROR_SUCCESS;
  }
  if (resource == "/gpu/dpcpp/occa") {
    match = "dpcpp";
    return CEED_ERROR_SUCCESS;
  }
  if (resource == "/gpu/opencl/occa") {
    match = "opencl";
    return CEED_ERROR_SUCCESS;
  }
  if (resource == "/cpu/openmp/occa") {
    match = "openmp";
    return CEED_ERROR_SUCCESS;
  }
  if (resource == "/cpu/self/occa") {
    match = "serial";
    return CEED_ERROR_SUCCESS;
  }

  // Skip initial slash
  for (int i = 1; i <= charCount; ++i) {
    const char c = c_resource[i];

    if (parsingStep == RESOURCE) {
      if (c == RESOURCE_DELIMITER || c == QUERY_DELIMITER || c == '\0') {
        resourceVector.push_back(resource.substr(wordStart, i - wordStart));
        wordStart = i + 1;

        // Check if we are done parsing the resource
        if (c == QUERY_DELIMITER) {
          parsingStep = QUERY_KEY;
        }
      }
    } else if (parsingStep == QUERY_KEY) {
      if (c == QUERY_KEY_VALUE_DELIMITER) {
        queryKey  = resource.substr(wordStart, i - wordStart);
        wordStart = i + 1;

        // Looking to parse the query value now
        parsingStep = QUERY_VALUE;
      }
    } else if (parsingStep == QUERY_VALUE) {
      if (c == QUERY_ARG_DELIMITER || c == '\0') {
        query[queryKey] = resource.substr(wordStart, i - wordStart);
        wordStart       = i + 1;

        // Back to parsing the next query argument
        parsingStep = QUERY_KEY;
        queryKey    = "";
      }
    }
  }

  // Looking for [match, "occa"]
  if (resourceVector.size() != 2 || resourceVector[1] != "occa") {
    return 1;
  }

  match = resourceVector[0];
  return CEED_ERROR_SUCCESS;
}

void setDefaultProps(::occa::properties &deviceProps, const std::string &defaultMode) {
  std::string mode;
  if (deviceProps.has("mode")) {
    // Don't override mode if passed
    mode = (std::string)deviceProps["mode"];
  } else {
    mode = defaultMode;
    deviceProps.set("mode", mode);
  }

  // Set default device id
  if ((mode == "CUDA") || (mode == "HIP") || (mode == "dpcpp") || (mode == "OpenCL")) {
    if (!deviceProps.has("device_id")) {
      deviceProps["device_id"] = 0;
    }
  }

  // Set default platform id
  if ((mode == "dpcpp") || (mode == "OpenCL")) {
    if (!deviceProps.has("platform_id")) {
      deviceProps["platform_id"] = 0;
    }
  }
}

static int initCeed(const char *c_resource, Ceed ceed) {
  int         ierr;
  std::string match;
  StringMap   query;

  ierr = splitCeedResource(c_resource, match, query);
  if (ierr) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "(OCCA) Backend cannot use resource: %s", c_resource);
  }

  std::string mode;
  ierr = getDeviceMode(match, mode);
  if (ierr) {
    return CeedError(ceed, CEED_ERROR_BACKEND, "(OCCA) Backend cannot use resource: %s", c_resource);
  }

  std::string               devicePropsStr = "{\n";
  StringMap::const_iterator it;
  for (it = query.begin(); it != query.end(); ++it) {
    devicePropsStr += "  \"";
    devicePropsStr += it->first;
    devicePropsStr += "\": ";
    devicePropsStr += it->second;
    devicePropsStr += ",\n";
  }
  devicePropsStr += '}';

  ::occa::properties deviceProps(devicePropsStr);
  setDefaultProps(deviceProps, mode);

  ceed::occa::Context *context = new Context(::occa::device(deviceProps));
  CeedCallBackend(CeedSetData(ceed, context));

  return CEED_ERROR_SUCCESS;
}

static int destroyCeed(Ceed ceed) {
  delete Context::from(ceed);
  return CEED_ERROR_SUCCESS;
}

static int registerCeedFunction(Ceed ceed, const char *fname, ceed::occa::ceedFunction f) {
  return CeedSetBackendFunction(ceed, "Ceed", ceed, fname, f);
}

static int preferHostMemType(CeedMemType *type) {
  *type = CEED_MEM_HOST;
  return CEED_ERROR_SUCCESS;
}

static int preferDeviceMemType(CeedMemType *type) {
  *type = CEED_MEM_DEVICE;
  return CEED_ERROR_SUCCESS;
}

static ceed::occa::ceedFunction getPreferredMemType(Ceed ceed) {
  if (Context::from(ceed)->device.hasSeparateMemorySpace()) {
    return (ceed::occa::ceedFunction)(void *)preferDeviceMemType;
  }
  return (ceed::occa::ceedFunction)(void *)preferHostMemType;
}

static int registerMethods(Ceed ceed) {
  CeedOccaRegisterBaseFunction("Destroy", ceed::occa::destroyCeed);
  CeedOccaRegisterBaseFunction("GetPreferredMemType", getPreferredMemType(ceed));
  CeedOccaRegisterBaseFunction("VectorCreate", ceed::occa::Vector::ceedCreate);
  CeedOccaRegisterBaseFunction("BasisCreateTensorH1", ceed::occa::TensorBasis::ceedCreate);
  CeedOccaRegisterBaseFunction("BasisCreateH1", ceed::occa::SimplexBasis::ceedCreate);
  CeedOccaRegisterBaseFunction("ElemRestrictionCreate", ceed::occa::ElemRestriction::ceedCreate);
  CeedOccaRegisterBaseFunction("ElemRestrictionCreateBlocked", ceed::occa::ElemRestriction::ceedCreateBlocked);
  CeedOccaRegisterBaseFunction("QFunctionCreate", ceed::occa::QFunction::ceedCreate);
  CeedOccaRegisterBaseFunction("QFunctionContextCreate", ceed::occa::QFunctionContext::ceedCreate);
  CeedOccaRegisterBaseFunction("OperatorCreate", ceed::occa::Operator::ceedCreate);
  CeedOccaRegisterBaseFunction("CompositeOperatorCreate", ceed::occa::Operator::ceedCreateComposite);

  return CEED_ERROR_SUCCESS;
}

static int registerBackend(const char *resource, Ceed ceed) {
  try {
    CeedCallBackend(ceed::occa::initCeed(resource, ceed));
  } catch (const ::occa::exception &e) {
    CeedHandleOccaException(e);
  }
  try {
    CeedCallBackend(ceed::occa::registerMethods(ceed));
  } catch (const ::occa::exception &e) {
    CeedHandleOccaException(e);
  }
  return CEED_ERROR_SUCCESS;
}
}  // namespace occa
}  // namespace ceed

CEED_INTERN int CeedRegister_Occa(void) {
  // General mode
  CeedCallBackend(CeedRegister("/*/occa", ceed::occa::registerBackend, 270));
  // CPU Modes
  CeedCallBackend(CeedRegister("/cpu/self/occa", ceed::occa::registerBackend, 260));
  CeedCallBackend(CeedRegister("/cpu/openmp/occa", ceed::occa::registerBackend, 250));
  // GPU Modes
  CeedCallBackend(CeedRegister("/gpu/dpcpp/occa", ceed::occa::registerBackend, 240));
  CeedCallBackend(CeedRegister("/gpu/opencl/occa", ceed::occa::registerBackend, 230));
  CeedCallBackend(CeedRegister("/gpu/hip/occa", ceed::occa::registerBackend, 220));
  CeedCallBackend(CeedRegister("/gpu/cuda/occa", ceed::occa::registerBackend, 210));
  return CEED_ERROR_SUCCESS;
}
