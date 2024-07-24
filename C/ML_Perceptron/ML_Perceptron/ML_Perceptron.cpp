#include <stdio.h>
#include <math.h>
/*
#define INPUT_SIZE 21
#define HIDDEN_NEURONS 2
#define OUTPUT_NEURONS 1
#define EPOCHS 10000
#define LEARNING_RATE 0.10000

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1 - x);
}

double X[INPUT_SIZE] = { 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0 };
double y[INPUT_SIZE] = { 0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0 };

int main()
{
    double weights1[HIDDEN_NEURONS] = { 1, 1 };
    double weights2[HIDDEN_NEURONS] = { 1, 0 };
    double bias1[HIDDEN_NEURONS] = { 1, 1 };
    double bias2[OUTPUT_NEURONS] = { 1 };

    //Training the network
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            double hidden_layer_input[HIDDEN_NEURONS] = { 0, 0 };
            double hidden_layer_output[HIDDEN_NEURONS] = { 0, 0};
            double output_layer_input = 0;
            double predicted_output = 0;

            //Calculate hidden layer input and output
            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                hidden_layer_input[j] = X[i] * weights1[j] + bias1[j];
                hidden_layer_output[j] = sigmoid(hidden_layer_input[j]);
            }

            //Calculate output layer input and output
            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                output_layer_input += hidden_layer_output[j] * weights2[j];
            }
            output_layer_input += bias2[0];
            predicted_output = sigmoid(output_layer_input);

            //Backprogpagarion
            double error = y[i] - predicted_output;
            double d_predicted_output = error * sigmoid_derivative(predicted_output);

            //Update weights and biases for the output layer
            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                weights2[j] += hidden_layer_output[j] * d_predicted_output * LEARNING_RATE;
            }
            bias2[0] += d_predicted_output * LEARNING_RATE;

            //Calculate error for the hidden layer
            double error_hidden_layer[HIDDEN_NEURONS] = { 0 };
            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                error_hidden_layer[j] = d_predicted_output * weights2[j];
            }

            //Update weights and biases for the hidden layer
            for (int j = 0; j < HIDDEN_NEURONS; j++)
            {
                double d_hidden_layer = error_hidden_layer[j] * sigmoid_derivative(hidden_layer_output[j]);
                weights1[j] += X[i] * d_hidden_layer * LEARNING_RATE;
                bias1[j] += d_hidden_layer * LEARNING_RATE;
            }

        }

    }


    double test_input[4] = { 0.3, 0.45, 0.6, 0.9 };
    printf("predicted output for input: \n");
    for (int i = 0; i < 4; i++)
    {
        double hidden_layer_input[HIDDEN_NEURONS] = { 0, 0 };
        double hidden_layer_output[HIDDEN_NEURONS] = { 0, 0 };
        double output_layer_input = 0;
        double predicted_output = 0;
        //Calculate hidden layer input and output for the test input
        for (int j = 0; j < HIDDEN_NEURONS; j++)
        {
            hidden_layer_input[j] = test_input[i] * weights1[j] + bias1[j];
            hidden_layer_output[j] = sigmoid(hidden_layer_input[j]);
        }

        //Calculate output layer input and output for the test input
        for (int j = 0; j < HIDDEN_NEURONS; j++)
        {
            output_layer_input += hidden_layer_output[j] * weights2[j];
        }
        output_layer_input += bias2[0];
        predicted_output = sigmoid(output_layer_input);

        printf("% f: % f\n", test_input[i], predicted_output);

    }
        printf("Done");
        return 0;
}

*/