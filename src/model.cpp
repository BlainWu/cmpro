#include "../include/cmpro/model.h"

Model::Model(Configure& config_) {
    status = NewSession(tensorflow::SessionOptions(), &session);
    std::ifstream fin;
    config = config_;
    model_path = config.MODEL_NAME;
    is_predicting = false;
    if(model_load()){
        std::cerr << "Fail to load model" << std::endl;
    }
    else{
        std::cerr << "Succeed to load model" << std::endl;

    }
}

int Model::model_load() {
    status = ReadBinaryProto(tensorflow::Env::Default(), "../model/"+model_path+".pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    return 0;
}

int Model::model_predict(std::vector<double> dataori) {
    if(is_predicting){
        return -1;
    }
    else{
        is_predicting = true;
    }
    int ans=0;
    tensorflow::Tensor tinput
    (tensorflow::DT_FLOAT, tensorflow::TensorShape({1, config.TENSOR_SHAPE}));
    assign("input", &dataori);

    std::vector<tensorflow::Tensor> outputs;
    auto dst = tinput.flat<float>().data();
    copy_n(dataori.begin(), config.TENSOR_SHAPE, dst);
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
            { "input", tinput}
    };
    status = session->Run(inputs, {"output"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;
        return -1;
    }
    std::cout << "Output tensor size:" << outputs.size() << std::endl;
    tensorflow::Tensor t = outputs[0];

    auto tmap = t.tensor<float, 2>();
    int output_class_id = -1;
    double output_prob = 0.0;
    for(int j=0;j<config.TENSOR_OUTPUT;++j){
        if (tmap(0, j) >= output_prob) {
            output_class_id = j;
            output_prob = tmap(0, j);
        }
    }
    ans = output_class_id;
    is_predicting = false;
    return ans;
}

void Model::assign(std::string tname, std::vector<double>* vec) {
    //Convert input 1-D double vector to Tensor
    int ndim = vec->size();
    if (ndim == 0) {
        std::cout << "WARNING: Input Vec size is 0 ..." << std::endl;
        return;
    }
    // Create New tensor and set value
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, ndim})); // New Tensor shape [1, ndim]
    auto x_map = x.tensor<float, 2>();
    for (int j = 0; j < ndim; j++) {
        x_map(0, j) = (*vec)[j];
    }
    // Append <tname, Tensor> to input
    input.emplace_back(tname, x);
}