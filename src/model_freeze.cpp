#include<cstdlib>
#include<string>
#include<cstdio>
#include<cstring>
#include<iostream>
#include<algorithm>

using namespace std;

const int N = 300;

void Test(){
    char line[N];
    FILE *fp;
    string stin;
    cout << "model name: " << endl;
    cin >> stin;
    string cmd;
    cmd = "python ../third_party/freeze_graph.py --input_graph=../third_party/models/"+stin+"/nn_model.pbtxt --input_checkpoint=../third_party/models/"+stin+"/nn_model.ckpt --output_graph=../model/"+stin+".pb --output_node_names=output";
    //引号内是你的linux指令
    // 系统调用
    const char *sysCommand = cmd.data();
    if ((fp = popen(sysCommand, "r")) == NULL) {
        cout << "error" << endl;
        return;
    }
    while (fgets(line, sizeof(line)-1, fp) != NULL){
        cout << line ;
    }
    pclose(fp);
}

int main(){
    Test();

    return 0;
}