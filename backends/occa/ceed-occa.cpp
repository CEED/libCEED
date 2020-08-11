// Copyright (c) 2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <map>
#include <vector>
#include <occa.hpp>

#include "ceed-occa-context.hpp"
#include "ceed-occa-elem-restriction.hpp"
#include "ceed-occa-operator.hpp"
#include "ceed-occa-qfunction.hpp"
#include "ceed-occa-simplex-basis.hpp"
#include "ceed-occa-tensor-basis.hpp"
#include "ceed-occa-types.hpp"
#include "ceed-occa-vector.hpp"


namespace ceed {
  namespace occa {
    typedef std::map<std::string, std::string> StringMap;
    typedef std::vector<std::string> StringVector;

    enum ResourceParserStep {
      RESOURCE,
      QUERY_KEY,
      QUERY_VALUE
    };

    static const char RESOURCE_DELIMITER = '/';
    static const char QUERY_DELIMITER = ':';
    static const char QUERY_KEY_VALUE_DELIMITER = '=';
    static const char QUERY_ARG_DELIMITER = ',';

    static std::string getDefaultDeviceMode(const bool cpuMode,
                                            const bool gpuMode) {
      // In case both cpuMode and gpuMode are set, prioritize the GPU if available
      // For example, if the resource is "/*/occa"
      if (gpuMode) {
        if (::occa::modeIsEnabled("CUDA")) {
          return "CUDA";
        }
        if (::occa::modeIsEnabled("HIP")) {
          return "HIP";
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

    static int getDeviceMode(const std::string &match,
                             std::string &mode) {
      if (match == "cuda") {
        mode = "CUDA";
        return 0;
      }
      if (match == "hip") {
        mode = "HIP";
        return 0;
      }
      if (match == "opencl") {
        mode = "OpenCL";
        return 0;
      }
      if (match == "openmp") {
        mode = "OpenMP";
        return 0;
      }
      if (match == "serial") {
        mode = "Serial";
        return 0;
      }

      const bool autoMode = match == "*";
      const bool cpuMode = match == "cpu";
      const bool gpuMode = match == "gpu";

      mode = getDefaultDeviceMode(cpuMode || autoMode,
                                  gpuMode || autoMode);
      return !mode.size();
    }

    static int splitCeedResource(const std::string &resource,
                                 std::string &match,
                                 StringMap &query) {
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
      const int charCount = (int) resource.size();
      const char *c_resource = resource.c_str();

      StringVector resourceVector;

      ResourceParserStep parsingStep = RESOURCE;
      int wordStart = 1;
      std::string queryKey;

      // Check for /gpu/occa/cuda, /gpu/occa/hip, /cpu/occa/serial, /cpu/occa/openmp
      // Note: added for matching style with other backends
      if (resource == "/gpu/occa/cuda"){
        match = "cuda";
        return 0;
      }
      if (resource == "/gpu/occa/hip"){
        match = "hip";
        return 0;
      }
      if (resource == "/cpu/occa/serial"){
        match = "serial";
        return 0;
      }
      if (resource == "/cpu/occa/openmp"){
        match = "openmp";
        return 0;
      }

      // Skip initial slash
      for (int i = 1; i <= charCount; ++i) {
        const char c = c_resource[i];

        if (parsingStep == RESOURCE) {
          if (c == RESOURCE_DELIMITER || c == QUERY_DELIMITER || c == '\0') {
            resourceVector.push_back(
              resource.substr(wordStart, i - wordStart)
            );
            wordStart = i + 1;

            // Check if we are done parsing the resource
            if (c == QUERY_DELIMITER) {
              parsingStep = QUERY_KEY;
            }
          }
        }
        else if (parsingStep == QUERY_KEY) {
          if (c == QUERY_KEY_VALUE_DELIMITER) {
            queryKey = resource.substr(wordStart, i - wordStart);
            wordStart = i + 1;

            // Looking to parse the query value now
            parsingStep = QUERY_VALUE;
          }
        } else if (parsingStep == QUERY_VALUE) {
          if (c == QUERY_ARG_DELIMITER || c == '\0') {
            query[queryKey] = resource.substr(wordStart, i - wordStart);
            wordStart = i + 1;

            // Back to parsing the next query argument
            parsingStep = QUERY_KEY;
            queryKey = "";
          }
        }
      }

      // Looking for [match, "occa"]
      if (resourceVector.size() != 2 || resourceVector[1] != "occa") {
        return 1;
      }

      match = resourceVector[0];
      return 0;
    }

    void setDefaultProps(::occa::properties &deviceProps,
                         const std::string &defaultMode) {
      std::string mode;
      if (deviceProps.has("mode")) {
        // Don't override mode if passed
        mode = (std::string) deviceProps["mode"];
      } else {
        mode = defaultMode;
        deviceProps["mode"] = mode;
      }

      // Set default device id
      if ((mode == "CUDA")
          || (mode == "HIP")
          || (mode == "OpenCL")) {
        if (!deviceProps.has("device_id")) {
          deviceProps["device_id"] = 0;
        }
      }

      // Set default platform id
      if (mode == "OpenCL") {
        if (!deviceProps.has("platform_id")) {
          deviceProps["platform_id"] = 0;
        }
      }
    }

    static int initCeed(const char *c_resource, Ceed ceed) {
      int ierr;
      std::string match;
      StringMap query;

      ierr = splitCeedResource(c_resource, match, query);
      if (ierr) {
        return CeedError(ceed, 1, "(OCCA) Backend cannot use resource: %s", c_resource);
      }

      std::string mode;
      ierr = getDeviceMode(match, mode);
      if (ierr) {
        return CeedError(ceed, 1, "(OCCA) Backend cannot use resource: %s", c_resource);
      }

      std::string devicePropsStr = "{\n";
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
      ierr = CeedSetData(ceed, (void**) &context); CeedChk(ierr);

      return 0;
    }

    static int destroyCeed(Ceed ceed) {
      delete Context::from(ceed);
      return 0;
    }

    static int registerCeedFunction(Ceed ceed, const char *fname, ceed::occa::ceedFunction f) {
      return CeedSetBackendFunction(ceed, "Ceed", ceed, fname, f);
    }

    static int preferHostMemType(CeedMemType *type) {
      *type = CEED_MEM_HOST;
      return 0;
    }

    static int preferDeviceMemType(CeedMemType *type) {
      *type = CEED_MEM_DEVICE;
      return 0;
    }

    static ceed::occa::ceedFunction getPreferredMemType(Ceed ceed) {
      if (Context::from(ceed)->device.hasSeparateMemorySpace()) {
        return (ceed::occa::ceedFunction) (void*) preferDeviceMemType;
      }
      return (ceed::occa::ceedFunction) (void*) preferHostMemType;
    }

    static int registerMethods(Ceed ceed) {
      int ierr;

      CeedOccaRegisterBaseFunction("Destroy", ceed::occa::destroyCeed);
      CeedOccaRegisterBaseFunction("GetPreferredMemType", getPreferredMemType(ceed));
      CeedOccaRegisterBaseFunction("VectorCreate", ceed::occa::Vector::ceedCreate);
      CeedOccaRegisterBaseFunction("BasisCreateTensorH1", ceed::occa::TensorBasis::ceedCreate);
      CeedOccaRegisterBaseFunction("BasisCreateH1", ceed::occa::SimplexBasis::ceedCreate);
      CeedOccaRegisterBaseFunction("ElemRestrictionCreate", ceed::occa::ElemRestriction::ceedCreate);
      CeedOccaRegisterBaseFunction("ElemRestrictionCreateBlocked", ceed::occa::ElemRestriction::ceedCreateBlocked);
      CeedOccaRegisterBaseFunction("QFunctionCreate", ceed::occa::QFunction::ceedCreate);
      CeedOccaRegisterBaseFunction("OperatorCreate", ceed::occa::Operator::ceedCreate);
      CeedOccaRegisterBaseFunction("CompositeOperatorCreate", ceed::occa::Operator::ceedCreateComposite);

      return 0;
    }

    static int registerBackend(const char *resource, Ceed ceed) {
      int ierr;

      try {
        ierr = ceed::occa::initCeed(resource, ceed); CeedChk(ierr);
        ierr = ceed::occa::registerMethods(ceed); CeedChk(ierr);
      } catch (::occa::exception &exc) {
        CeedHandleOccaException(exc);
      }

      return 0;
    }
  }
}

__attribute__((constructor))
static void Register(void) {
  CeedRegister("/*/occa", ceed::occa::registerBackend, 80);
  CeedRegister("/cpu/occa", ceed::occa::registerBackend, 70);
  CeedRegister("/serial/occa", ceed::occa::registerBackend, 60);
  CeedRegister("/cpu/occa/serial", ceed::occa::registerBackend, 60);
  CeedRegister("/openmp/occa", ceed::occa::registerBackend, 50);
  CeedRegister("/cpu/occa/openmp", ceed::occa::registerBackend, 50);
  CeedRegister("/opencl/occa", ceed::occa::registerBackend, 40);
  CeedRegister("/gpu/occa", ceed::occa::registerBackend, 30);
  CeedRegister("/gpu/occa/hip", ceed::occa::registerBackend, 20);
  CeedRegister("/hip/occa", ceed::occa::registerBackend, 20);
  CeedRegister("/gpu/occa/cuda", ceed::occa::registerBackend, 10);
  CeedRegister("/cuda/occa", ceed::occa::registerBackend, 10);

}
