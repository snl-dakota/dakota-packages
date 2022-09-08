#include "MUQ/Modeling/UMBridge/UMBridgeModPiece.h"

using namespace muq::Modeling;


UMBridgeModPiece::UMBridgeModPiece(const std::string host, json config, httplib::Headers headers)
: config(config), client(host, headers),
  ModPiece(read_input_size(host, headers), read_output_size(host, headers))
{
  this->outputs.resize(this->numOutputs);
}


Eigen::VectorXi UMBridgeModPiece::read_input_size(const std::string host, const httplib::Headers& headers){
  // Would prefer to reuse the existing client, circular dependency in constructor though...
  umbridge::HTTPModel client(host, headers);
  return StdVectorToEigenvectori(client.inputSizes);
}

Eigen::VectorXi UMBridgeModPiece::read_output_size(const std::string host, const httplib::Headers& headers){
  umbridge::HTTPModel client(host, headers);
  return StdVectorToEigenvectori(client.outputSizes);
}

void UMBridgeModPiece::EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs) {
  if (!client.SupportsEvaluate())
    throw std::runtime_error("Model does not support evaluation!");
  std::vector<std::vector<double>> inputs_stdvec(this->numInputs);
  for (int i = 0; i < this->numInputs; i++) {
    inputs_stdvec[i] = EigenvectordToStdVector(inputs[i]);
  }
  client.Evaluate(inputs_stdvec, config);
  for (int i = 0; i < this->numOutputs; i++) {
    outputs[i] = StdVectorToEigenvectord(client.outputs[i]);
  }
}

void UMBridgeModPiece::GradientImpl(unsigned int outWrt,
                                unsigned int inWrt,
                                muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                Eigen::VectorXd const& sens) {
  if (client.SupportsGradient()) {
    client.Gradient(outWrt, inWrt, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(sens), config);
    gradient = StdVectorToEigenvectord(client.gradient);
  } else
    gradient = GradientByFD(outWrt, inWrt, inputs, sens);
}

void UMBridgeModPiece::ApplyJacobianImpl(unsigned int outWrt,
                                    unsigned int inWrt,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                    Eigen::VectorXd const& vec){
  if (client.SupportsApplyJacobian()) {
    client.ApplyJacobian(outWrt, inWrt, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(vec), config);
    jacobianAction = StdVectorToEigenvectord(client.jacobianAction);
  } else
    jacobianAction = ApplyJacobianByFD(outWrt, inWrt, inputs, vec);
}

void UMBridgeModPiece::ApplyHessianImpl(unsigned int outWrt,
                                    unsigned int inWrt1,
                                    unsigned int inWrt2,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& inputs,
                                    Eigen::VectorXd const& sens,
                                    Eigen::VectorXd const& vec){
  if (client.SupportsApplyHessian()) {
    client.ApplyHessian(outWrt, inWrt1, inWrt2, EigenvectordsToStdVectors(inputs), EigenvectordToStdVector(sens), EigenvectordToStdVector(vec), config);
    hessAction = StdVectorToEigenvectord(client.hessAction);
  } else
    hessAction = ApplyHessianByFD(outWrt, inWrt1, inWrt2, inputs, sens, vec);
}