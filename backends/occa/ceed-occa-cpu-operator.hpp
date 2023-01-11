// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#ifndef CEED_OCCA_CPU_OPERATOR_HEADER
#define CEED_OCCA_CPU_OPERATOR_HEADER

#include <sstream>
#include <vector>

#include "ceed-occa-operator.hpp"
#include "ceed-occa-vector.hpp"

namespace ceed {
namespace occa {
class Basis;
class SimplexBasis;
class TensorBasis;

class CpuOperator : public Operator {
 private:
  typedef std::vector<Vector *> VectorVector;
  typedef std::vector<Basis *>  BasisVector;

  VectorVector dofInputs, dofOutputs;

 public:
  CpuOperator();

  ~CpuOperator();

  // Setup helper vectors
  void setupVectors();

  void setupVectors(const int fieldCount, OperatorFieldVector &opFields, QFunctionFieldVector &qfFields, VectorVector &vectors);

  void freeVectors();

  // Restriction operators
  void setupInputs(Vector *in);

  void setupOutputs(Vector *out);

  void applyQFunction();

  // Push arguments for a given field
  void pushKernelArgs(Vector *vec, const bool isInput, const int index);

  void pushTensorBasisKernelArgs(const QFunctionField &qfField, TensorBasis &basis);

  void pushSimplexBasisKernelArgs(const QFunctionField &qfField, SimplexBasis &basis);

  // Set props for a given field
  ::occa::properties getKernelProps();

  void applyAdd(Vector *in, Vector *out);

  ::occa::kernel buildApplyAddKernel();

  //---[ Kernel Generation ]------------------
  void addBasisFunctionSource(std::stringstream &ss);

  void addBasisIfMissingSource(BasisVector &sourceBasis, Basis *basis);

  void addKernelSource(std::stringstream &ss);

  void addKernelArgsSource(std::stringstream &ss);

  void addKernelArgSource(std::stringstream &ss, const bool isInput, const int index);

  void addTensorKernelArgSource(std::stringstream &ss, const bool isInput, const int index, const OperatorField &opField,
                                const QFunctionField &qfField, std::stringstream &dimAttribute);

  void addSimplexKernelArgSource(std::stringstream &ss, const bool isInput, const int index, const OperatorField &opField,
                                 const QFunctionField &qfField, std::stringstream &dimAttribute);

  void addQuadArraySource(std::stringstream &ss);

  void addSingleQfunctionQuadArraySource(std::stringstream &ss, const bool isInput, const int index, const std::string &name);

  void addQfunctionQuadArraySource(std::stringstream &ss, const bool isInput, const int count, const std::string &name);

  void addInputSetupSource(std::stringstream &ss);

  void addOutputSetupSource(std::stringstream &ss);

  void addBasisApplySource(std::stringstream &ss, const bool isInput, const int count);

  void addInterpSource(std::stringstream &ss, const bool isInput, const int index);

  void addGradTensorSource(std::stringstream &ss, const bool isInput, const int index);

  void addGradSimplexSource(std::stringstream &ss, const bool isInput, const int index);

  void addWeightSource(std::stringstream &ss, const bool isInput, const int index);

  void addCopySource(std::stringstream &ss, const bool isInput, const int index);

  void addQFunctionApplicationSource(std::stringstream &ss);

  //  ---[ Variables ]---------------
  inline std::string xputName(const bool isInput) { return isInput ? "input" : "output"; }

  inline std::string indexedVar(const std::string &name, const int index) { return name + std::to_string(index); }

  inline std::string indexedVar(const std::string &name, const bool isInput, const int index) {
    return (isInput ? "input" : "output") + std::to_string(index) + "_" + name;
  }

  inline std::string dofInputVar(const int index) { return indexedVar("dofInput", index); }

  inline std::string dofOutputVar(const int index) { return indexedVar("dofOutput", index); }

  inline std::string interpVar(const bool isInput, const int index) { return indexedVar("B", isInput, index); }

  inline std::string gradVar(const bool isInput, const int index) { return indexedVar("Bx", isInput, index); }

  inline std::string qWeightVar(const bool isInput, const int index) { return indexedVar("qWeights", isInput, index); }

  std::string elementFunction(const bool isInput, const int index);

  std::string fieldFunctionName(const QFunctionField &qfField);

  std::string fullFieldFunctionName(const bool isInput, const OperatorField &opField, const QFunctionField &qfField);
};
}  // namespace occa
}  // namespace ceed

#endif
