import numpy as np

tab = "    "

def print_matrix_mul(data, inputs_size, output_size, input_name, output_name):
    s = ""
    for i in range(output_size):
        s += f"{tab}{output_name}[{i}] = tanhf({data[i * inputs_size]}f * {input_name}[0]"
        for j in range(1, inputs_size):
            s += f" + {data[i * inputs_size + j]}f * {input_name}[{j}]"

        s += f" + {data[output_size * inputs_size + i]}f);\n"

    return s

def print_network(genotype, layers):
    layer_names = ["inputs"]
    s = ""
    for i in range(1, len(layers) - 1):
        layer_names.append(f"temp_{i}")
        s += f"static float temp_{i}[{layers[i]}];\n"
    layer_names.append("outputs")

    s += f"\nvoid forward(float inputs[{layers[0]}], float outputs[{layers[-1]}]) {{"
    for i in range(len(layer_names) - 1):
        s += "\n"
        s += print_matrix_mul(genotype, layers[i], layers[i+1], layer_names[i], layer_names[i+1])
        genotype = genotype[layers[i] * layers[i + 1] + layers[i + 1]:]

    return s

with open("genotype.csv", "r") as file:
    genotype = file.read().split()

genotype = np.array([float(num) for num in genotype])

with open("layers.csv", "r") as file:
    layers = file.read().split(",")

layers = np.array([int(num) for num in layers])

with open("neural_network.c", "w") as file:
    file.write(f"#include <math.h>\n\n")
    s = print_network(genotype, layers)
    file.write(s)
    file.write("}\n")

with open("neural_network.h", "w") as file:
    file.write("#pragma once\n")
    file.write(f"\nvoid forward(float inputs[{layers[0]}], float outputs[{layers[-1]}]);\n")
