#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    double* weights;
    double* weight_gradients;
    int num_inputs;
    double bias;
    double bias_gradient;
    double output;
} Neuron;

typedef struct {
    Neuron* neurons;
    int num_neurons;
} Layer;

typedef struct {
    Layer layers[10];
    int num_layers;
    double learning_rate;
    double* momentum;
    int batch_size;
    int current_batch;
} myAI;

myAI* createMyAI(int* layer_sizes, int num_layers, double learning_rate, int batch_size);
void destroyMyAI(myAI* ai);
void forward(myAI* ai, double* inputs);
void backward(myAI* ai, double* inputs, double* targets);
void updateWeights(myAI* ai);
void saveMyAI(myAI* ai, const char* filename);
myAI* loadMyAI(const char* filename);
int getLayerCount(myAI* ai);
int getNeuronCount(myAI* ai, int layer_index);
double getLearningRate(myAI* ai);
void setLearningRate(myAI* ai, double new_learning_rate);
void resetGradients(myAI* ai);
double calculateError(myAI* ai, double* target, int num_targets);



double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1 - x);
}

myAI* createMyAI(int* layer_sizes, int num_layers, double learning_rate, int batch_size) {
    myAI* ai = (myAI*)malloc(sizeof(myAI));
    ai->num_layers = num_layers;
    ai->learning_rate = learning_rate;
    ai->batch_size = batch_size;
    ai->current_batch = 0;

    for (int i = 0; i < num_layers; i++) {
        ai->layers[i].num_neurons = layer_sizes[i];
        ai->layers[i].neurons = (Neuron*)malloc(layer_sizes[i] * sizeof(Neuron));

        for (int j = 0; j < layer_sizes[i]; j++) {
            ai->layers[i].neurons[j].num_inputs = (i == 0) ? 0 : layer_sizes[i-1];
            ai->layers[i].neurons[j].weights = (double*)malloc(ai->layers[i].neurons[j].num_inputs * sizeof(double));
            ai->layers[i].neurons[j].weight_gradients = (double*)calloc(ai->layers[i].neurons[j].num_inputs, sizeof(double));
            ai->layers[i].neurons[j].bias = ((double)rand() / RAND_MAX) * 2 - 1;
            ai->layers[i].neurons[j].bias_gradient = 0;

            for (int k = 0; k < ai->layers[i].neurons[j].num_inputs; k++) {
                ai->layers[i].neurons[j].weights[k] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }

    // Initialize momentum
    int total_weights = 0;
    for (int i = 1; i < num_layers; i++) {
        total_weights += layer_sizes[i] * layer_sizes[i-1] + layer_sizes[i];
    }
    ai->momentum = (double*)calloc(total_weights, sizeof(double));

    return ai;
}

void destroyMyAI(myAI* ai) {
    for (int i = 0; i < ai->num_layers; i++) {
        for (int j = 0; j < ai->layers[i].num_neurons; j++) {
            free(ai->layers[i].neurons[j].weights);
            free(ai->layers[i].neurons[j].weight_gradients);
        }
        free(ai->layers[i].neurons);
    }
    free(ai->momentum);
    free(ai);
}

void forward(myAI* ai, double* inputs) {
    for (int i = 0; i < ai->num_layers; i++) {
        double* current_inputs = (i == 0) ? inputs : (double*)malloc(ai->layers[i-1].num_neurons * sizeof(double));
        
        if (i > 0) {
            for (int j = 0; j < ai->layers[i-1].num_neurons; j++) {
                current_inputs[j] = ai->layers[i-1].neurons[j].output;
            }
        }

        for (int j = 0; j < ai->layers[i].num_neurons; j++) {
            double sum = 0;
            for (int k = 0; k < ai->layers[i].neurons[j].num_inputs; k++) {
                sum += current_inputs[k] * ai->layers[i].neurons[j].weights[k];
            }
            sum += ai->layers[i].neurons[j].bias;
            ai->layers[i].neurons[j].output = sigmoid(sum);
        }

        if (i > 0) {
            free(current_inputs);
        }
    }
}

void backward(myAI* ai, double* inputs, double* targets) {
    // Calculate output layer errors
    Layer* output_layer = &ai->layers[ai->num_layers - 1];
    double* output_errors = (double*)malloc(output_layer->num_neurons * sizeof(double));

    for (int i = 0; i < output_layer->num_neurons; i++) {
        double output = output_layer->neurons[i].output;
        output_errors[i] = (targets[i] - output) * sigmoid_derivative(output);
    }

    // Backpropagate errors and accumulate gradients
    for (int i = ai->num_layers - 1; i > 0; i--) {
        Layer* current_layer = &ai->layers[i];
        Layer* prev_layer = &ai->layers[i - 1];

        double* prev_layer_errors = (double*)malloc(prev_layer->num_neurons * sizeof(double));
        memset(prev_layer_errors, 0, prev_layer->num_neurons * sizeof(double));

        for (int j = 0; j < current_layer->num_neurons; j++) {
            Neuron* neuron = &current_layer->neurons[j];
            double error = output_errors[j];

            for (int k = 0; k < neuron->num_inputs; k++) {
                neuron->weight_gradients[k] += error * prev_layer->neurons[k].output;
                prev_layer_errors[k] += error * neuron->weights[k];
            }
            neuron->bias_gradient += error;
        }

        free(output_errors);
        output_errors = prev_layer_errors;

        // Apply sigmoid derivative to the errors of the previous layer
        for (int j = 0; j < prev_layer->num_neurons; j++) {
            output_errors[j] *= sigmoid_derivative(prev_layer->neurons[j].output);
        }
    }

    free(output_errors);

    ai->current_batch++;
    if (ai->current_batch >= ai->batch_size) {
        updateWeights(ai);
        ai->current_batch = 0;
    }
}

void updateWeights(myAI* ai) {
    int momentum_index = 0;
    double learning_rate = ai->learning_rate / ai->batch_size;
    double momentum_factor = 0.9;

    for (int i = 1; i < ai->num_layers; i++) {
        Layer* layer = &ai->layers[i];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];
            for (int k = 0; k < neuron->num_inputs; k++) {
                double gradient = neuron->weight_gradients[k];
                ai->momentum[momentum_index] = momentum_factor * ai->momentum[momentum_index] + (1 - momentum_factor) * gradient;
                neuron->weights[k] += learning_rate * ai->momentum[momentum_index];
                neuron->weight_gradients[k] = 0; // Reset gradient
                momentum_index++;
            }
            double bias_gradient = neuron->bias_gradient;
            ai->momentum[momentum_index] = momentum_factor * ai->momentum[momentum_index] + (1 - momentum_factor) * bias_gradient;
            neuron->bias += learning_rate * ai->momentum[momentum_index];
            neuron->bias_gradient = 0; // Reset gradient
            momentum_index++;
        }
    }
}

void saveMyAI(myAI* ai, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    fwrite(&ai->num_layers, sizeof(int), 1, file);
    fwrite(&ai->learning_rate, sizeof(double), 1, file);
    fwrite(&ai->batch_size, sizeof(int), 1, file);

    for (int i = 0; i < ai->num_layers; i++) {
        fwrite(&ai->layers[i].num_neurons, sizeof(int), 1, file);
        for (int j = 0; j < ai->layers[i].num_neurons; j++) {
            fwrite(&ai->layers[i].neurons[j].num_inputs, sizeof(int), 1, file);
            fwrite(ai->layers[i].neurons[j].weights, sizeof(double), ai->layers[i].neurons[j].num_inputs, file);
            fwrite(&ai->layers[i].neurons[j].bias, sizeof(double), 1, file);
        }
    }

    fclose(file);
}

myAI* loadMyAI(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file for reading\n");
        return NULL;
    }

    int num_layers;
    double learning_rate;
    int batch_size;

    fread(&num_layers, sizeof(int), 1, file);
    fread(&learning_rate, sizeof(double), 1, file);
    fread(&batch_size, sizeof(int), 1, file);

    int* layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        fread(&layer_sizes[i], sizeof(int), 1, file);
    }

    myAI* ai = createMyAI(layer_sizes, num_layers, learning_rate, batch_size);

    for (int i = 0; i < num_layers; i++) {
        for (int j = 0; j < ai->layers[i].num_neurons; j++) {
            fread(&ai->layers[i].neurons[j].num_inputs, sizeof(int), 1, file);
            fread(ai->layers[i].neurons[j].weights, sizeof(double), ai->layers[i].neurons[j].num_inputs, file);
            fread(&ai->layers[i].neurons[j].bias, sizeof(double), 1, file);
        }
    }

    free(layer_sizes);
    fclose(file);
    return ai;
}

int getLayerCount(myAI* ai) {
    return ai->num_layers;
}

int getNeuronCount(myAI* ai, int layer_index) {
    if (layer_index >= 0 && layer_index < ai->num_layers) {
        return ai->layers[layer_index].num_neurons;
    }
    return 0;
}

double getLearningRate(myAI* ai) {
    return ai->learning_rate;
}

void setLearningRate(myAI* ai, double new_learning_rate) {
    ai->learning_rate = new_learning_rate;
}

void resetGradients(myAI* ai) {
    for (int i = 1; i < ai->num_layers; i++) {
        Layer* layer = &ai->layers[i];
        for (int j = 0; j < layer->num_neurons; j++) {
            Neuron* neuron = &layer->neurons[j];
            memset(neuron->weight_gradients, 0, neuron->num_inputs * sizeof(double));
            neuron->bias_gradient = 0;
        }
    }
    ai->current_batch = 0;
}

double calculateError(myAI* ai, double* target, int num_targets) {
    if (ai->num_layers == 0 || num_targets != ai->layers[ai->num_layers-1].num_neurons) {
        return 0.0;
    }
    
    double error = 0.0;
    Layer* output_layer = &ai->layers[ai->num_layers-1];
    for (int i = 0; i < num_targets; i++) {
        error += pow(target[i] - output_layer->neurons[i].output, 2);
    }
    return error / 2.0;
}
