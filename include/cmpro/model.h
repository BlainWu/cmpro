// model.model_load
// result = model.pridict(shape)
// shape : vector<int>


#ifndef SPMAT_MODEL_H
#define SPMAT_MODEL_H

#include <iostream>
#include <fstream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "configure.h"

class Model {
public:
    Model(Configure& config_);
    Configure config;
    int model_predict(std::vector<double> dataori);
private:
    int model_load();
    bool is_predicting;
    void assign(std::string, std::vector<double>*);
    std::vector<std::pair<std::string, tensorflow::Tensor> > input;
    tensorflow::Session* session;
    tensorflow::Status status;
    tensorflow::GraphDef graph_def;
    std::string model_path;
};


#endif //SPMAT_MODEL_H
