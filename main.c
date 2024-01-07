#include "neural_network.h"
#include "stdio.h"

int main() {
    float inputs[3072] = {0.f};
    float outputs[2] = {0.f};

    forward(inputs, outputs);
    
    for (int i = 0; i < 2; i++) {
        printf("output %f\n", outputs[i]);
    }

    return 0;
}
