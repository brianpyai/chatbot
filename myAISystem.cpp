#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "myAI.h"
#include "mydata.h"
#define SAMPLE_LENGTH 100
#define M_PI 3.14159265358979323846

// 新增示例函数声明
void music_genre_classifier();
void tic_tac_toe_ai();
void text_summarizer();
void speech_recognition_simulator();
void anomaly_detection_system();
// 示例函数声明
void text_classifier();
void stock_trading_system();
void recommendation_system();
void language_detector();
void weather_prediction();
void image_compression();
void reinforcement_learning_maze_solver();
void neural_network_compression();
void advanced_text_analyzer();
void multi_language_translation();
void intelligent_qa_system();
// 辅助函数
double random_double(double min, double max);
void generate_random_text(const char* language, myChar& text);

// 新增示例函数声明
void music_genre_classifier();
void tic_tac_toe_ai();
void text_summarizer();
// 在文件开头添加以下函数声明
void sentiment_analysis();
void autonomous_vehicle_navigation();
void energy_consumption_predictor();
void anomaly_detection_system();

// 主函数
int main() {
    srand(time(NULL));

    int choice;
    do {
        printf("\nMyAI Advanced Examples Menu:\n");
        printf("1. Simple Text Classifier\n");
        printf("2. Stock Trading System Simulation\n");
        printf("3. Recommendation System\n");
        printf("4. Language Detector\n");
        printf("5. Weather Prediction Model\n");
        printf("6. Image Compression (Simplified)\n");
        printf("7. Music Genre Classifier\n");
        printf("8. Tic-Tac-Toe AI\n");
        printf("9. Text Summarizer\n");
        printf("10. Speech Recognition Simulator\n");
        printf("11. Anomaly Detection System\n");
        printf("12. Neural Network Compression and Knowledge Distillation\n");
        printf("13. Sentiment Analysis for Social Media\n");
        printf("14. Autonomous Vehicle Navigation Simulator\n");
        printf("15. Energy Consumption Predictor\n");
    printf("161. Intelligent Q&A System\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch(choice) {
            case 1: text_classifier(); break;
            case 2: stock_trading_system(); break;
            case 3: recommendation_system(); break;
            case 4: language_detector(); break;
            case 5: weather_prediction(); break;
            case 6: image_compression(); break;
            case 7: music_genre_classifier(); break;
            case 8: tic_tac_toe_ai(); break;
            case 9: text_summarizer(); break;
            case 10: speech_recognition_simulator(); break;
            case 11: anomaly_detection_system(); break;
            case 12: neural_network_compression(); break;
            case 13: sentiment_analysis(); break;
            case 14: autonomous_vehicle_navigation(); break;
            case 15: energy_consumption_predictor(); break;
        case 16: intelligent_qa_system(); break;
            case 0: printf("Exiting...\n"); break;
            default: printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 0);

    return 0;
}
#define QA_VOCAB_SIZE 5000
#define QA_MAX_SENTENCE_LENGTH 100
#define QA_HIDDEN_SIZE 128



void intelligent_qa_system() {
    printf("Intelligent Q&A System\n");

    myDict knowledge_base;
    knowledge_base.set("What is the capital of France?", "The capital of France is Paris.");
    knowledge_base.set("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare.");
    knowledge_base.set("What is the boiling point of water?", "The boiling point of water is 100 degrees Celsius at sea level.");

    // Create a simple neural network for answer relevance scoring
    int layer_sizes[] = {QA_VOCAB_SIZE * 2, QA_HIDDEN_SIZE, 1};  // Input: question + answer, Output: relevance score
    myAI* relevance_scorer = createMyAI(layer_sizes, 3, 0.01, 32);

    // Train the relevance scorer (simplified training)
    for (int epoch = 0; epoch < 1000; epoch++) {
        double input[QA_VOCAB_SIZE * 2] = {0};
        double target[1];

        // Generate random "question" and "answer"
        for (int i = 0; i < QA_VOCAB_SIZE; i++) {
            input[i] = (rand() % 10 == 0) ? 1 : 0;  // Sparse representation
            input[QA_VOCAB_SIZE + i] = (rand() % 10 == 0) ? 1 : 0;
        }
        target[0] = (double)rand() / RAND_MAX;  // Random relevance score

        forward(relevance_scorer, input);
        backward(relevance_scorer, input, target);
    }

    // Q&A loop
    char question[QA_MAX_SENTENCE_LENGTH];
    while (1) {
        printf("Ask a question (or type 'exit' to quit): ");
        fgets(question, sizeof(question), stdin);
        question[strcspn(question, "\n")] = 0;  // Remove newline

        if (strcmp(question, "exit") == 0) {
            break;
        }

        const char* exact_answer = knowledge_base.get(question);
        if (exact_answer != NULL) {
            printf("Answer: %s\n", exact_answer);
        } else {
            printf("I don't have an exact answer, but here are some relevant information:\n");

            // Find most relevant answers
            double max_relevance = 0;
            const char* most_relevant_answer = NULL;

            for (auto it = knowledge_base.begin_items(); it != knowledge_base.end_items(); ++it) {
                auto [kb_question, kb_answer] = *it;

                // Create input for relevance scorer
                double input[QA_VOCAB_SIZE * 2] = {0};
                for (int i = 0; question[i] != '\0' && i < QA_VOCAB_SIZE; i++) {
                    input[question[i] % QA_VOCAB_SIZE] = 1;
                }
                for (int i = 0; kb_answer[i] != '\0' && i < QA_VOCAB_SIZE; i++) {
                    input[QA_VOCAB_SIZE + (kb_answer[i] % QA_VOCAB_SIZE)] = 1;
                }

                forward(relevance_scorer, input);
                double relevance = relevance_scorer->layers[relevance_scorer->num_layers-1].neurons[0].output;

                if (relevance > max_relevance) {
                    max_relevance = relevance;
                    most_relevant_answer = kb_answer;
                }
            }

            if (most_relevant_answer != NULL) {
                printf("%s (Relevance: %.2f)\n", most_relevant_answer, max_relevance);
            } else {
                printf("Sorry, I couldn't find any relevant information.\n");
            }
        }
    }

    destroyMyAI(relevance_scorer);
}

#define SENTIMENT_VOCAB_SIZE 1000
#define SENTIMENT_HIDDEN_SIZE 50
#define SENTIMENT_OUTPUT_SIZE 3  // Positive, Neutral, Negative

void sentiment_analysis() {
    printf("Sentiment Analysis for Social Media\n");

    int layer_sizes[] = {SENTIMENT_VOCAB_SIZE, SENTIMENT_HIDDEN_SIZE, SENTIMENT_OUTPUT_SIZE};
    myAI* sentiment_classifier = createMyAI(layer_sizes, 3, 0.01, 32);

    // Training data simulation
    for (int epoch = 0; epoch < 1000; epoch++) {
        double input[SENTIMENT_VOCAB_SIZE] = {0};
        double target[SENTIMENT_OUTPUT_SIZE] = {0};

        // Simulate a random social media post
        for (int i = 0; i < 20; i++) {
            int word_index = rand() % SENTIMENT_VOCAB_SIZE;
            input[word_index] = 1;
        }

        // Simulate sentiment label
        int sentiment = rand() % SENTIMENT_OUTPUT_SIZE;
        target[sentiment] = 1;

        forward(sentiment_classifier, input);
        backward(sentiment_classifier, input, target);

        if (epoch % 100 == 0) {
            double error = calculateError(sentiment_classifier, target, SENTIMENT_OUTPUT_SIZE);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // Test the sentiment classifier
    printf("Testing Sentiment Classifier:\n");
    const char* sentiments[] = {"Positive", "Neutral", "Negative"};
    for (int i = 0; i < 5; i++) {
        double test_input[SENTIMENT_VOCAB_SIZE] = {0};
        for (int j = 0; j < 20; j++) {
            int word_index = rand() % SENTIMENT_VOCAB_SIZE;
            test_input[word_index] = 1;
        }

        forward(sentiment_classifier, test_input);

        int predicted_sentiment = 0;
        double max_output = sentiment_classifier->layers[sentiment_classifier->num_layers-1].neurons[0].output;
        for (int j = 1; j < SENTIMENT_OUTPUT_SIZE; j++) {
            if (sentiment_classifier->layers[sentiment_classifier->num_layers-1].neurons[j].output > max_output) {
                max_output = sentiment_classifier->layers[sentiment_classifier->num_layers-1].neurons[j].output;
                predicted_sentiment = j;
            }
        }

        printf("Test %d: Predicted Sentiment = %s\n", i + 1, sentiments[predicted_sentiment]);
    }

    destroyMyAI(sentiment_classifier);
}

#define NAV_INPUT_SIZE 10  // e.g., distance to obstacles, current speed, etc.
#define NAV_HIDDEN_SIZE 20
#define NAV_OUTPUT_SIZE 3  // Left, Straight, Right

void autonomous_vehicle_navigation() {
    printf("Autonomous Vehicle Navigation Simulator\n");

    int layer_sizes[] = {NAV_INPUT_SIZE, NAV_HIDDEN_SIZE, NAV_OUTPUT_SIZE};
    myAI* nav_ai = createMyAI(layer_sizes, 3, 0.01, 32);

    // Training simulation
    for (int episode = 0; episode < 1000; episode++) {
        double total_reward = 0;
        for (int step = 0; step < 100; step++) {
            double input[NAV_INPUT_SIZE];
            for (int i = 0; i < NAV_INPUT_SIZE; i++) {
                input[i] = (double)rand() / RAND_MAX;
            }

            forward(nav_ai, input);

            int action = 0;
            double max_output = nav_ai->layers[nav_ai->num_layers-1].neurons[0].output;
            for (int i = 1; i < NAV_OUTPUT_SIZE; i++) {
                if (nav_ai->layers[nav_ai->num_layers-1].neurons[i].output > max_output) {
                    max_output = nav_ai->layers[nav_ai->num_layers-1].neurons[i].output;
                    action = i;
                }
            }

            // Simulate environment response and calculate reward
            double reward = (double)rand() / RAND_MAX - 0.3;  // Random reward with slight penalty
            total_reward += reward;

            // Create target for training
            double target[NAV_OUTPUT_SIZE] = {0};
            target[action] = reward;

            backward(nav_ai, input, target);
        }

        if (episode % 100 == 0) {
            printf("Episode %d, Total Reward: %f\n", episode, total_reward);
        }
    }

    // Test the navigation AI
    printf("Testing Autonomous Navigation:\n");
    const char* actions[] = {"Left", "Straight", "Right"};
    for (int i = 0; i < 5; i++) {
        double test_input[NAV_INPUT_SIZE];
        for (int j = 0; j < NAV_INPUT_SIZE; j++) {
            test_input[j] = (double)rand() / RAND_MAX;
        }

        forward(nav_ai, test_input);

        int action = 0;
        double max_output = nav_ai->layers[nav_ai->num_layers-1].neurons[0].output;
        for (int j = 1; j < NAV_OUTPUT_SIZE; j++) {
            if (nav_ai->layers[nav_ai->num_layers-1].neurons[j].output > max_output) {
                max_output = nav_ai->layers[nav_ai->num_layers-1].neurons[j].output;
                action = j;
            }
        }

        printf("Test %d: Action = %s\n", i + 1, actions[action]);
    }

    destroyMyAI(nav_ai);
}

#define ENERGY_INPUT_SIZE 7  // e.g., time of day, day of week, temperature, etc.
#define ENERGY_HIDDEN_SIZE 15
#define ENERGY_OUTPUT_SIZE 1

void energy_consumption_predictor() {
    printf("Energy Consumption Predictor\n");

    int layer_sizes[] = {ENERGY_INPUT_SIZE, ENERGY_HIDDEN_SIZE, ENERGY_OUTPUT_SIZE};
    myAI* energy_predictor = createMyAI(layer_sizes, 3, 0.001, 32);

    // Training simulation
    for (int epoch = 0; epoch < 10000; epoch++) {
        double input[ENERGY_INPUT_SIZE];
        double target[ENERGY_OUTPUT_SIZE];

        // Generate random input (normalize to 0-1 range)
        for (int i = 0; i < ENERGY_INPUT_SIZE; i++) {
            input[i] = (double)rand() / RAND_MAX;
        }

        // Simulate energy consumption (kWh) based on input
        target[0] = 0;
        for (int i = 0; i < ENERGY_INPUT_SIZE; i++) {
            target[0] += input[i] * (rand() % 10);  // Random weights
        }
        target[0] = target[0] / ENERGY_INPUT_SIZE * 100;  // Scale to realistic kWh values

        forward(energy_predictor, input);
        backward(energy_predictor, input, target);

        if (epoch % 1000 == 0) {
            double error = calculateError(energy_predictor, target, ENERGY_OUTPUT_SIZE);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // Test the energy consumption predictor
    printf("Testing Energy Consumption Predictor:\n");
    for (int i = 0; i < 5; i++) {
        double test_input[ENERGY_INPUT_SIZE];
        for (int j = 0; j < ENERGY_INPUT_SIZE; j++) {
            test_input[j] = (double)rand() / RAND_MAX;
        }

        forward(energy_predictor, test_input);

        double predicted_energy = energy_predictor->layers[energy_predictor->num_layers-1].neurons[0].output;

        printf("Test %d: Predicted Energy Consumption = %.2f kWh\n", i + 1, predicted_energy);
    }

    destroyMyAI(energy_predictor);
}

// 神经网络压缩和知识蒸馏
#define INPUT_SIZE 784 // 28x28 MNIST digits
#define HIDDEN_SIZE_LARGE 128
#define HIDDEN_SIZE_SMALL 32
#define OUTPUT_SIZE 10
#define NUM_SAMPLES 1000

void neural_network_compression() {
    printf("Neural Network Compression and Knowledge Distillation\n");

    // Create a large teacher network
    int teacher_layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE_LARGE, HIDDEN_SIZE_LARGE, OUTPUT_SIZE};
    myAI* teacher = createMyAI(teacher_layer_sizes, 4, 0.01, 32);

    // Create a smaller student network
    int student_layer_sizes[] = {INPUT_SIZE, HIDDEN_SIZE_SMALL, OUTPUT_SIZE};
    myAI* student = createMyAI(student_layer_sizes, 3, 0.01, 32);

    // Generate synthetic training data
    double** train_data = (double**)malloc(NUM_SAMPLES * sizeof(double*));
    double** train_labels = (double**)malloc(NUM_SAMPLES * sizeof(double*));
    for (int i = 0; i < NUM_SAMPLES; i++) {
        train_data[i] = (double*)malloc(INPUT_SIZE * sizeof(double));
        train_labels[i] = (double*)calloc(OUTPUT_SIZE, sizeof(double));
        for (int j = 0; j < INPUT_SIZE; j++) {
            train_data[i][j] = random_double(0, 1);
        }
        int label = rand() % OUTPUT_SIZE;
        train_labels[i][label] = 1.0;
    }

    // Train the teacher network
    printf("Training the teacher network...\n");
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_error = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            forward(teacher, train_data[i]);
            backward(teacher, train_data[i], train_labels[i]);
            total_error += calculateError(teacher, train_labels[i], OUTPUT_SIZE);
        }
        if (epoch % 10 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / NUM_SAMPLES);
        }
    }

    // Perform knowledge distillation
    printf("Performing knowledge distillation...\n");
    double temperature = 2.0;
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_error = 0.0;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            // Get soft targets from the teacher
            forward(teacher, train_data[i]);
            double soft_targets[OUTPUT_SIZE];
            double sum = 0.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                soft_targets[j] = exp(teacher->layers[teacher->num_layers-1].neurons[j].output / temperature);
                sum += soft_targets[j];
            }
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                soft_targets[j] /= sum;
            }

            // Train the student using soft targets
            forward(student, train_data[i]);
            backward(student, train_data[i], soft_targets);
            total_error += calculateError(student, soft_targets, OUTPUT_SIZE);
        }
        if (epoch % 10 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / NUM_SAMPLES);
        }
    }

    // Compare teacher and student performance
    printf("Comparing teacher and student performance:\n");
    int teacher_correct = 0, student_correct = 0;
    for (int i = 0; i < NUM_SAMPLES; i++) {
        forward(teacher, train_data[i]);
        int teacher_prediction = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (teacher->layers[teacher->num_layers-1].neurons[j].output >
                teacher->layers[teacher->num_layers-1].neurons[teacher_prediction].output) {
                teacher_prediction = j;
            }
        }

        forward(student, train_data[i]);
        int student_prediction = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (student->layers[student->num_layers-1].neurons[j].output >
                student->layers[student->num_layers-1].neurons[student_prediction].output) {
                student_prediction = j;
            }
        }

        int true_label = 0;
        for (int j = 1; j < OUTPUT_SIZE; j++) {
            if (train_labels[i][j] > train_labels[i][true_label]) {
                true_label = j;
            }
        }

        if (teacher_prediction == true_label) teacher_correct++;
        if (student_prediction == true_label) student_correct++;
    }

    printf("Teacher accuracy: %.2f%%\n", 100.0 * teacher_correct / NUM_SAMPLES);
    printf("Student accuracy: %.2f%%\n", 100.0 * student_correct / NUM_SAMPLES);
    printf("Compression ratio: %.2f\n", (double)(INPUT_SIZE * HIDDEN_SIZE_LARGE * 2 + HIDDEN_SIZE_LARGE * OUTPUT_SIZE) /
                                        (INPUT_SIZE * HIDDEN_SIZE_SMALL + HIDDEN_SIZE_SMALL * OUTPUT_SIZE));

    // Clean up
    for (int i = 0; i < NUM_SAMPLES; i++) {
        free(train_data[i]);
        free(train_labels[i]);
    }
    free(train_data);
    free(train_labels);
    destroyMyAI(teacher);
    destroyMyAI(student);
}



// 新增示例函数实现
void music_genre_classifier() {
    printf("Music Genre Classifier\n");

    #define NUM_GENRES 5
    #define NUM_FEATURES 10

    const char* genres[] = {"Rock", "Pop", "Classical", "Jazz", "Electronic"};

    int layer_sizes[] = {NUM_FEATURES, 20, NUM_GENRES};
    myAI* classifier = createMyAI(layer_sizes, 3, 0.01, 32);

    // 训练分类器
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < 100; i++) {  // 每个epoch训练100次
            double input[NUM_FEATURES];
            double target[NUM_GENRES] = {0};

            // 生成随机的"音乐特征"
            for (int j = 0; j < NUM_FEATURES; j++) {
                input[j] = random_double(0, 1);
            }

            int target_genre = rand() % NUM_GENRES;
            target[target_genre] = 1.0;

            forward(classifier, input);
            backward(classifier, input, target);

            total_error += calculateError(classifier, target, NUM_GENRES);
        }

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / 100);
        }
    }

    // 测试分类器
    printf("Testing the music genre classifier:\n");
    for (int i = 0; i < 5; i++) {
        double test_input[NUM_FEATURES];
        for (int j = 0; j < NUM_FEATURES; j++) {
            test_input[j] = random_double(0, 1);
        }

        forward(classifier, test_input);

        int predicted_genre = 0;
        double max_output = classifier->layers[classifier->num_layers - 1].neurons[0].output;
        for (int j = 1; j < NUM_GENRES; j++) {
            if (classifier->layers[classifier->num_layers - 1].neurons[j].output > max_output) {
                max_output = classifier->layers[classifier->num_layers - 1].neurons[j].output;
                predicted_genre = j;
            }
        }

        printf("Test %d: Predicted Genre = %s\n", i + 1, genres[predicted_genre]);
    }

    destroyMyAI(classifier);
}

void tic_tac_toe_ai() {
    printf("Tic-Tac-Toe AI\n");

    #define BOARD_SIZE 9

    int layer_sizes[] = {BOARD_SIZE, 18, BOARD_SIZE};
    myAI* ai = createMyAI(layer_sizes, 3, 0.01, 1);

    // 训练AI
    for (int epoch = 0; epoch < 10000; epoch++) {
        double board[BOARD_SIZE] = {0};
        double target[BOARD_SIZE] = {0};

        // 模拟一个随机的游戏状态
        int num_moves = rand() % BOARD_SIZE;
        for (int i = 0; i < num_moves; i++) {
            int move;
            do {
                move = rand() % BOARD_SIZE;
            } while (board[move] != 0);
            board[move] = (i % 2 == 0) ? 1 : -1;
        }

        // 生成目标（合法的下一步移动）
        int num_valid_moves = 0;
        for (int i = 0; i < BOARD_SIZE; i++) {
            if (board[i] == 0) {
                target[i] = 1;
                num_valid_moves++;
            }
        }
        for (int i = 0; i < BOARD_SIZE; i++) {
            target[i] /= num_valid_moves;  // 归一化目标
        }

        forward(ai, board);
        backward(ai, board, target);

        if (epoch % 1000 == 0) {
            double error = calculateError(ai, target, BOARD_SIZE);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // 测试AI
    printf("Testing Tic-Tac-Toe AI:\n");
    double test_board[BOARD_SIZE] = {1, 0, -1, 0, 1, -1, 0, 0, 0};
    printf("Board state:\n");
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i % 3 == 0) printf("\n");
        if (test_board[i] == 1) printf("X ");
        else if (test_board[i] == -1) printf("O ");
        else printf("- ");
    }
    printf("\n");

    forward(ai, test_board);

    printf("AI's move probabilities:\n");
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i % 3 == 0) printf("\n");
        printf("%.2f ", ai->layers[ai->num_layers - 1].neurons[i].output);
    }
    printf("\n");

    destroyMyAI(ai);
}

void text_summarizer() {
    printf("Text Summarizer\n");

    #define VOCAB_SIZE 1000
    #define MAX_SENTENCE_LENGTH 20
    #define NUM_SENTENCES 10

    int layer_sizes[] = {VOCAB_SIZE * MAX_SENTENCE_LENGTH, 100, MAX_SENTENCE_LENGTH};
    myAI* summarizer = createMyAI(layer_sizes, 3, 0.001, 32);

    // 创建一个简单的词汇表
    myDict vocabulary;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        char word[20];
        snprintf(word, sizeof(word), "word%d", i);
        vocabulary.set(word, "1");
    }

    // 训练摘要生成器
    for (int epoch = 0; epoch < 10000; epoch++) {
        double input[VOCAB_SIZE * MAX_SENTENCE_LENGTH] = {0};
        double target[MAX_SENTENCE_LENGTH] = {0};

        // 生成随机的"文档"
        for (int i = 0; i < NUM_SENTENCES; i++) {
            int sentence_length = rand() % MAX_SENTENCE_LENGTH + 1;
            for (int j = 0; j < sentence_length; j++) {
                int word_index = rand() % VOCAB_SIZE;
                input[i * MAX_SENTENCE_LENGTH + j] = 1;
            }
            if (i < MAX_SENTENCE_LENGTH) {
                target[i] = 1.0 / MAX_SENTENCE_LENGTH;  // 假设每个句子都有相等的概率被选为摘要
            }
        }

        forward(summarizer, input);
        backward(summarizer, input, target);

        if (epoch % 1000 == 0) {
            double error = calculateError(summarizer, target, MAX_SENTENCE_LENGTH);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // 测试摘要生成器
    printf("Testing Text Summarizer:\n");
    double test_input[VOCAB_SIZE * MAX_SENTENCE_LENGTH] = {0};
    for (int i = 0; i < NUM_SENTENCES; i++) {
        int sentence_length = rand() % MAX_SENTENCE_LENGTH + 1;
        for (int j = 0; j < sentence_length; j++) {
            int word_index = rand() % VOCAB_SIZE;
            test_input[i * MAX_SENTENCE_LENGTH + j] = 1;
        }
    }

    forward(summarizer, test_input);

    printf("Sentence importance scores:\n");
    for (int i = 0; i < MAX_SENTENCE_LENGTH; i++) {
        printf("Sentence %d: %.4f\n", i + 1, summarizer->layers[summarizer->num_layers - 1].neurons[i].output);
    }

    destroyMyAI(summarizer);
}

void speech_recognition_simulator() {
    printf("Speech Recognition Simulator\n");

    #define NUM_FEATURES 13  // 模拟MFCC特征
    #define NUM_FRAMES 50
    #define NUM_WORDS 10

    const char* words[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"};

    int layer_sizes[] = {NUM_FEATURES * NUM_FRAMES, 100, NUM_WORDS};
    myAI* recognizer = createMyAI(layer_sizes, 3, 0.001, 32);

    // 训练识别器
    for (int epoch = 0; epoch < 10000; epoch++) {
        double input[NUM_FEATURES * NUM_FRAMES];
        double target[NUM_WORDS] = {0};

        // 生成随机的"音频特征"
        for (int i = 0; i < NUM_FEATURES * NUM_FRAMES; i++) {
            input[i] = random_double(0, 1);
        }

        int target_word = rand() % NUM_WORDS;
        target[target_word] = 1.0;

        forward(recognizer, input);
        backward(recognizer, input, target);

        if (epoch % 1000 == 0) {
            double error = calculateError(recognizer, target, NUM_WORDS);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // 测试识别器
    printf("Testing Speech Recognition Simulator:\n");
    for (int i = 0; i < 5; i++) {
        double test_input[NUM_FEATURES * NUM_FRAMES];
        for (int j = 0; j < NUM_FEATURES * NUM_FRAMES; j++) {
            test_input[j] = random_double(0, 1);
        }

        forward(recognizer, test_input);

        int recognized_word = 0;
        double max_output = recognizer->layers[recognizer->num_layers - 1].neurons[0].output;
        for (int j = 1; j < NUM_WORDS; j++) {
            if (recognizer->layers[recognizer->num_layers - 1].neurons[j].output > max_output) {
                max_output = recognizer->layers[recognizer->num_layers - 1].neurons[j].output;
                recognized_word = j;
            }
        }

        printf("Test %d: Recognized Word = %s\n", i + 1, words[recognized_word]);
    }

    destroyMyAI(recognizer);
}

void anomaly_detection_system() {
    printf("Anomaly Detection System\n");

    #define NUM_FEATURES 5
    #define THRESHOLD 0.1

    int layer_sizes[] = {NUM_FEATURES, 3, NUM_FEATURES};
    myAI* autoencoder = createMyAI(layer_sizes, 3, 0.01, 32);

    // 训练自动编码器
    for (int epoch = 0; epoch < 10000; epoch++) {
        double input[NUM_FEATURES];
        
        // 生成正常数据
        for (int i = 0; i < NUM_FEATURES; i++) {
            input[i] = random_double(0.4, 0.6);  // 正常数据集中在0.4-0.6之间
        }

        forward(autoencoder, input);
        backward(autoencoder, input, input);  // 自动编码器的目标是重构输入

        if (epoch % 1000 == 0) {
            double error = calculateError(autoencoder, input, NUM_FEATURES);
            printf("Epoch %d, Error: %f\n", epoch, error);
        }
    }

    // 测试异常检测系统
    printf("Testing Anomaly Detection System:\n");
    for (int i = 0; i < 10; i++) {
        double test_input[NUM_FEATURES];
        bool is_anomaly = (rand() % 2 == 0);  // 随机生成正常或异常数据

        if (is_anomaly) {
            // 生成异常数据
            for (int j = 0; j < NUM_FEATURES; j++) {
                test_input[j] = random_double(0, 1);
            }
        } else {
            // 生成正常数据
            for (int j = 0; j < NUM_FEATURES; j++) {
                test_input[j] = random_double(0.4, 0.6);
            }
        }

        forward(autoencoder, test_input);

        double reconstruction_error = 0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            double diff = test_input[j] - autoencoder->layers[autoencoder->num_layers - 1].neurons[j].output;
            reconstruction_error += diff * diff;
        }
        reconstruction_error /= NUM_FEATURES;

        bool detected_anomaly = (reconstruction_error > THRESHOLD);

        printf("Test %d: ", i + 1);
        printf("Actual: %s, ", is_anomaly ? "Anomaly" : "Normal");
        printf("Detected: %s, ", detected_anomaly ? "Anomaly" : "Normal");
        printf("Reconstruction Error: %f\n", reconstruction_error);
    }

    destroyMyAI(autoencoder);
}
void text_classifier() {
    printf("Simple Text Classifier\n");

    #define VOCAB_SIZE 1000
    #define NUM_CATEGORIES 3
    #define NUM_SAMPLES 1000

    int layer_sizes[] = {VOCAB_SIZE, 50, NUM_CATEGORIES};
    myAI* classifier = createMyAI(layer_sizes, 3, 0.01, 32);

    // 创建一个词汇表
    myDict vocabulary;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        char word[20];
        snprintf(word, sizeof(word), "word%d", i);
        vocabulary.set(word, "1");
    }

    // 训练分类器
    for (int epoch = 0; epoch < 100; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < NUM_SAMPLES; i++) {
            // 生成随机文本
            myDict word_count;
            int text_length = rand() % 50 + 10;  // 10 到 59 个词
            for (int j = 0; j < text_length; j++) {
                char word[20];
                snprintf(word, sizeof(word), "word%d", rand() % VOCAB_SIZE);
                const char* count = word_count.get(word);
                if (count == nullptr) {
                    word_count.set(word, "1");
                } else {
                    int new_count = atoi(count) + 1;
                    char count_str[10];
                    snprintf(count_str, sizeof(count_str), "%d", new_count);
                    word_count.set(word, count_str);
                }
            }

            // 将文本转换为输入向量
            double input[VOCAB_SIZE] = {0};
            for (auto it = word_count.begin_items(); it != word_count.end_items(); ++it) {
                auto [word, count] = *it;
                if (vocabulary.get(word) != nullptr) {
                    int index = atoi(vocabulary.get(word)) - 1;
                    input[index] = atof(count) / text_length;
                }
            }

            // 生成随机目标类别
            int target_category = rand() % NUM_CATEGORIES;
            double target[NUM_CATEGORIES] = {0};
            target[target_category] = 1.0;

            // 训练网络
            forward(classifier, input);
            backward(classifier, input, target);

            total_error += calculateError(classifier, target, NUM_CATEGORIES);
        }

        if (epoch % 10 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / NUM_SAMPLES);
        }
    }

    // 测试分类器
    printf("Testing the classifier:\n");
    for (int i = 0; i < 5; i++) {
        myDict word_count;
        int text_length = rand() % 50 + 10;
        for (int j = 0; j < text_length; j++) {
            char word[20];
            snprintf(word, sizeof(word), "word%d", rand() % VOCAB_SIZE);
            const char* count = word_count.get(word);
            if (count == nullptr) {
                word_count.set(word, "1");
            } else {
                int new_count = atoi(count) + 1;
                char count_str[10];
                snprintf(count_str, sizeof(count_str), "%d", new_count);
                word_count.set(word, count_str);
            }
        }

        double input[VOCAB_SIZE] = {0};
        for (auto it = word_count.begin_items(); it != word_count.end_items(); ++it) {
            auto [word, count] = *it;
            if (vocabulary.get(word) != nullptr) {
                int index = atoi(vocabulary.get(word)) - 1;
                input[index] = atof(count) / text_length;
            }
        }

        forward(classifier, input);

        int predicted_category = 0;
        double max_output = classifier->layers[classifier->num_layers - 1].neurons[0].output;
        for (int j = 1; j < NUM_CATEGORIES; j++) {
            if (classifier->layers[classifier->num_layers - 1].neurons[j].output > max_output) {
                max_output = classifier->layers[classifier->num_layers - 1].neurons[j].output;
                predicted_category = j;
            }
        }

        printf("Sample %d - Predicted category: %d\n", i + 1, predicted_category);
    }

    destroyMyAI(classifier);
}





#define STOCK_NUM_STOCKS 100
#define STOCK_INPUT_SIZE (STOCK_NUM_STOCKS * 2)
#define STOCK_HIDDEN_SIZE 400
#define STOCK_OUTPUT_SIZE STOCK_NUM_STOCKS
#define STOCK_MAX_DAYS 365

// 函數聲明
void update_stock_prices(myDict& stock_data, double* input);
void generate_training_targets(double* target);
void train_ai(myAI* ai, double* input, double* target);
void execute_trades(myAI* ai, myDict& stock_data, myDict& portfolio, double& balance);
void update_portfolio(myDict& portfolio, const char* stock_name, int shares_change);
double calculate_total_assets(myDict& stock_data, myDict& portfolio, double balance);
void print_final_portfolio(myDict& stock_data, myDict& portfolio, double balance);

void stock_trading_system() {
    printf("開始股票交易系統模擬\n");

    int training_days = 100;
    printf("請輸入要訓練嘅日數 (預設100): ");
    if (scanf("%d", &training_days) != 1 || training_days < 0) {
        printf("輸入無效，使用預設值 100 日\n");
        training_days = 100;
    }

    // 初始化 AI 模型
    printf("初始化 AI 模型\n");
    int layer_sizes[] = {STOCK_INPUT_SIZE, STOCK_HIDDEN_SIZE, STOCK_OUTPUT_SIZE};
    myAI* trading_ai = createMyAI(layer_sizes, 3, 0.001, 32);
    if (trading_ai == NULL) {
        printf("創建 AI 模型失敗\n");
        return;
    }
    printf("AI 模型創建成功\n");

    // 初始化股票數據同投資組合
    printf("初始化股票數據同投資組合\n");
    myDict stock_data;
    myDict portfolio;
    double balance = 100000.0;

    double* input = (double*)malloc(STOCK_INPUT_SIZE * sizeof(double));
    double* target = (double*)malloc(STOCK_OUTPUT_SIZE * sizeof(double));
    if (input == NULL || target == NULL) {
        printf("內存分配失敗\n");
        free(input);
        free(target);
        destroyMyAI(trading_ai);
        return;
    }
    printf("內存分配成功\n");

    // 初始化股票數據
    for (int i = 0; i < STOCK_NUM_STOCKS; i++) {
        char stock_name[20], price_str[20];
        snprintf(stock_name, sizeof(stock_name), "STOCK%d", i);
        snprintf(price_str, sizeof(price_str), "%.2f", 50.0 + (rand() % 5000) / 100.0);
        stock_data.set(stock_name, price_str);
    }

    // 訓練階段
    printf("開始訓練 AI 模型...\n");
    for (int day = 0; day < training_days; day++) {
        update_stock_prices(stock_data, input);
        generate_training_targets(target);
        train_ai(trading_ai, input, target);

        if (day % 100 == 0) {
            printf("訓練進度: %d/%d 日\n", day, training_days);
        }
    }
    printf("AI 模型訓練完成\n");

    // 重置股票數據同投資組合
    stock_data = myDict();
    portfolio = myDict();
    balance = 100000.0;
    for (int i = 0; i < STOCK_NUM_STOCKS; i++) {
        char stock_name[20], price_str[20];
        snprintf(stock_name, sizeof(stock_name), "STOCK%d", i);
        snprintf(price_str, sizeof(price_str), "%.2f", 50.0 + (rand() % 5000) / 100.0);
        stock_data.set(stock_name, price_str);
    }

    // 模擬交易階段
    printf("開始模擬交易...\n");
    for (int day = 0; day < STOCK_MAX_DAYS; day++) {
        update_stock_prices(stock_data, input);
        forward(trading_ai, input);
        execute_trades(trading_ai, stock_data, portfolio, balance);

        if (day % 365 == 0) {
            printf("第 %d 年結束，餘額: %.2f\n", day / 365, balance);
        }
    }

    // 輸出最終結果
    printf("模擬結束，最終結果：\n");
    print_final_portfolio(stock_data, portfolio, balance);

    // 清理資源
    printf("清理資源\n");
    free(input);
    free(target);
    destroyMyAI(trading_ai);

    printf("程序正常結束\n");
}

void update_stock_prices(myDict& stock_data, double* input) {
    int index = 0;
    for (int i = 0; i < STOCK_NUM_STOCKS; i++) {
        char stock_name[20], new_price_str[20];
        snprintf(stock_name, sizeof(stock_name), "STOCK%d", i);
        const char* price_str = stock_data.get(stock_name);
        if (price_str == NULL) continue;

        double old_price = atof(price_str);
        double new_price = old_price * (1 + ((rand() % 11) - 5) / 100.0);
        double volume = 10000 + (rand() % 90001);

        snprintf(new_price_str, sizeof(new_price_str), "%.2f", new_price);
        stock_data.set(stock_name, new_price_str);

        if (index + 1 < STOCK_INPUT_SIZE) {
            input[index++] = new_price / 100.0;
            input[index++] = volume / 100000.0;
        }
    }
}

void generate_training_targets(double* target) {
    for (int i = 0; i < STOCK_OUTPUT_SIZE; i++) {
        target[i] = (rand() % 101) / 100.0;
    }
}

void train_ai(myAI* ai, double* input, double* target) {
    forward(ai, input);
    backward(ai, input, target);
}

void execute_trades(myAI* ai, myDict& stock_data, myDict& portfolio, double& balance) {
    for (int i = 0; i < STOCK_NUM_STOCKS; i++) {
        if (i >= ai->layers[ai->num_layers - 1].num_neurons) {
            printf("錯誤：神經元索引超出範圍\n");
            continue;
        }
        double decision = ai->layers[ai->num_layers - 1].neurons[i].output;
        char stock_name[20];
        snprintf(stock_name, sizeof(stock_name), "STOCK%d", i);
        const char* price_str = stock_data.get(stock_name);
        if (price_str == NULL) continue;
        double price = atof(price_str);

        if (decision > 0.6 && balance > price) {
            int shares_to_buy = (int)(balance * 0.1 / price);
            if (shares_to_buy > 0) {
                balance -= shares_to_buy * price;
                update_portfolio(portfolio, stock_name, shares_to_buy);
            }
        } else if (decision < 0.4) {
            const char* shares_str = portfolio.get(stock_name);
            if (shares_str) {
                int shares = atoi(shares_str);
                int shares_to_sell = shares / 2;
                if (shares_to_sell > 0) {
                    balance += shares_to_sell * price;
                    update_portfolio(portfolio, stock_name, -shares_to_sell);
                }
            }
        }
    }
}

void update_portfolio(myDict& portfolio, const char* stock_name, int shares_change) {
    const char* current_shares_str = portfolio.get(stock_name);
    int current_shares = current_shares_str ? atoi(current_shares_str) : 0;
    int new_shares = current_shares + shares_change;
    
    char new_shares_str[20];
    snprintf(new_shares_str, sizeof(new_shares_str), "%d", new_shares);
    portfolio.set(stock_name, new_shares_str);
}

double calculate_total_assets(myDict& stock_data, myDict& portfolio, double balance) {
    double total_assets = balance;
    for (myDict::ItemIterator it = portfolio.begin_items(); it != portfolio.end_items(); ++it) {
        std::pair<const char*, const char*> item = *it;
        const char* stock_name = item.first;
        int shares = atoi(item.second);
        const char* price_str = stock_data.get(stock_name);
        if (price_str) {
            double price = atof(price_str);
            total_assets += shares * price;
        }
    }
    return total_assets;
}

void print_final_portfolio(myDict& stock_data, myDict& portfolio, double balance) {
    printf("最終投資組合:\n");
    for (myDict::ItemIterator it = portfolio.begin_items(); it != portfolio.end_items(); ++it) {
        try {
            std::pair<const char*, const char*> item = *it;
            if (item.first && item.second) {
                const char* stock_name = item.first;
                int shares = atoi(item.second);
                const char* price_str = stock_data.get(stock_name);
                if (price_str) {
                    double price = atof(price_str);
                    printf("%s: %d 股, 當前價格: %.2f\n", stock_name, shares, price);
                } else {
                    printf("%s: %d 股, 價格未知\n", stock_name, shares);
                }
            }
        } catch (const std::exception& e) {
            printf("錯誤: %s\n", e.what());
            break;
        }
    }
    printf("現金餘額: %.2f\n", balance);
    double total_assets = calculate_total_assets(stock_data, portfolio, balance);
    printf("總資產: %.2f\n", total_assets);
}


void recommendation_system() {
    printf("Recommendation System\n");

    #define NUM_USERS 1000
    #define NUM_ITEMS 500
    #define NUM_FEATURES 20

    int layer_sizes[] = {NUM_FEATURES * 2, 50, 1};
    myAI* recommender = createMyAI(layer_sizes, 3, 0.001, 32);

    // 创建用户和物品的特征向量
    double user_features[NUM_USERS][NUM_FEATURES];
    double item_features[NUM_ITEMS][NUM_FEATURES];
    for (int i = 0; i < NUM_USERS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            user_features[i][j] = random_double(0, 1);
        }
    }
    for (int i = 0; i < NUM_ITEMS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            item_features[i][j] = random_double(0, 1);
        }
    }

    // 训练推荐系统
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < 1000; i++) {  // 每个epoch训练1000次
            int user = rand() % NUM_USERS;
            int item = rand() % NUM_ITEMS;

            double input[NUM_FEATURES * 2];
            memcpy(input, user_features[user], NUM_FEATURES * sizeof(double));
            memcpy(input + NUM_FEATURES, item_features[item], NUM_FEATURES * sizeof(double));

            // 生成一个模拟的用户评分（0到5之间）
            double target = random_double(0, 5);

            forward(recommender, input);
            backward(recommender, input, &target);

            total_error += calculateError(recommender, &target, 1);
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / 1000);
        }
    }

    // 测试推荐系统
    printf("Testing the recommendation system:\n");
    for (int i = 0; i < 5; i++) {
        int user = rand() % NUM_USERS;
        int item = rand() % NUM_ITEMS;

        double input[NUM_FEATURES * 2];
        memcpy(input, user_features[user], NUM_FEATURES * sizeof(double));
        memcpy(input + NUM_FEATURES, item_features[item], NUM_FEATURES * sizeof(double));

        forward(recommender, input);
        double predicted_rating = recommender->layers[recommender->num_layers - 1].neurons[0].output * 5;  // 缩放到0-5范围

        printf("User %d, Item %d: Predicted Rating = %.2f\n", user, item, predicted_rating);
    }

    destroyMyAI(recommender);
}

void language_detector() {
    printf("Language Detector\n");

    #define NUM_LANGUAGES 5
    #define VOCAB_SIZE 1000
    #define SAMPLE_LENGTH 100

    const char* languages[] = {"English", "Spanish", "French", "German", "Italian"};

    int layer_sizes[] = {VOCAB_SIZE, 50, NUM_LANGUAGES};
    myAI* detector = createMyAI(layer_sizes, 3, 0.01, 32);

    // 创建一个简单的词汇表
    myDict vocabulary;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        char word[20];
        snprintf(word, sizeof(word), "word%d", i);
        vocabulary.set(word, "1");
    }

    // 训练语言检测器
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_error = 0.0;

        for (int i = 0; i < 100; i++) {  // 每个epoch训练100次
            int target_language = rand() % NUM_LANGUAGES;
            
            myChar text;
            generate_random_text(languages[target_language], text);

            // 将文本转换为输入向量
            double input[VOCAB_SIZE] = {0};
            myDict word_count;
            char* token = strtok(text.chars, " ");
            while (token != NULL) {
                const char* count = word_count.get(token);
                if (count == nullptr) {
                    word_count.set(token, "1");
                } else {
                    int new_count = atoi(count) + 1;
                    char count_str[10];
                    snprintf(count_str, sizeof(count_str), "%d", new_count);
                    word_count.set(token, count_str);
                }
                token = strtok(NULL, " ");
            }

            for (auto it = word_count.begin_items(); it != word_count.end_items(); ++it) {
                auto [word, count] = *it;
                if (vocabulary.get(word) != nullptr) {
                    int index = atoi(vocabulary.get(word)) - 1;
                    input[index] = atof(count) / SAMPLE_LENGTH;
                }
            }

            double target[NUM_LANGUAGES] = {0};
            target[target_language] = 1.0;

            forward(detector, input);
            backward(detector, input, target);

            total_error += calculateError(detector, target, NUM_LANGUAGES);
        }

        if (epoch % 100 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / 100);
        }
    }

    // 测试语言检测器
    printf("Testing the language detector:\n");
    for (int i = 0; i < 5; i++) {
        int target_language = rand() % NUM_LANGUAGES;
        
        myChar text;
        generate_random_text(languages[target_language], text);

        double input[VOCAB_SIZE] = {0};
        myDict word_count;
        char* token = strtok(text.chars, " ");
        while (token != NULL) {
            const char* count = word_count.get(token);
            if (count == nullptr) {
                word_count.set(token, "1");
            } else {
                int new_count = atoi(count) + 1;
                char count_str[10];
                snprintf(count_str, sizeof(count_str), "%d", new_count);
                word_count.set(token, count_str);
            }
            token = strtok(NULL, " ");
        }

        for (auto it = word_count.begin_items(); it != word_count.end_items(); ++it) {
            auto [word, count] = *it;
            if (vocabulary.get(word) != nullptr) {
                int index = atoi(vocabulary.get(word)) - 1;
                input[index] = atof(count) / SAMPLE_LENGTH;
            }
        }

        forward(detector, input);

        int detected_language = 0;
        double max_output = detector->layers[detector->num_layers - 1].neurons[0].output;
        for (int j = 1; j < NUM_LANGUAGES; j++) {
            if (detector->layers[detector->num_layers - 1].neurons[j].output > max_output) {
                max_output = detector->layers[detector->num_layers - 1].neurons[j].output;
                detected_language = j;
            }
        }

        printf("Actual: %s, Detected: %s\n", languages[target_language], languages[detected_language]);
    }

    destroyMyAI(detector);
}

void weather_prediction() {
    printf("Weather Prediction Model\n");

    #define NUM_FEATURES 5  // Temperature, Humidity, Pressure, Wind Speed, Precipitation
    #define NUM_DAYS 7

    int layer_sizes[] = {NUM_FEATURES * NUM_DAYS, 20, NUM_FEATURES};
    myAI* weather_model = createMyAI(layer_sizes, 3, 0.001, 32);

    // 训练天气预测模型
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_error = 0.0;

        double input[NUM_FEATURES * NUM_DAYS];
        double target[NUM_FEATURES];

        // 生成随机的天气数据序列
        for (int i = 0; i < NUM_FEATURES * NUM_DAYS; i++) {
            input[i] = random_double(0, 1);  // 假设所有特征都已归一化到0-1范围
        }

        // 生成目标（下一天的天气）
        for (int i = 0; i < NUM_FEATURES; i++) {
            // 使用简单的规则来生成下一天的天气（这里只是一个示例，实际预测会更复杂）
            target[i] = (input[i] + input[i + NUM_FEATURES]) / 2 + random_double(-0.1, 0.1);
            if (target[i] < 0) target[i] = 0;
            if (target[i] > 1) target[i] = 1;
        }

        forward(weather_model, input);
        backward(weather_model, input, target);

        total_error += calculateError(weather_model, target, NUM_FEATURES);

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Error: %f\n", epoch, total_error);
        }
    }

    // 测试天气预测模型
    printf("Testing the weather prediction model:\n");
    double test_input[NUM_FEATURES * NUM_DAYS];
    for (int i = 0; i < NUM_FEATURES * NUM_DAYS; i++) {
        test_input[i] = random_double(0, 1);
    }

    forward(weather_model, test_input);

    printf("Current weather (last day of input):\n");
    for (int i = NUM_FEATURES * (NUM_DAYS - 1); i < NUM_FEATURES * NUM_DAYS; i++) {
        printf("Feature %d: %.2f\n", i % NUM_FEATURES + 1, test_input[i]);
    }

    printf("\nPredicted weather for tomorrow:\n");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("Feature %d: %.2f\n", i + 1, weather_model->layers[weather_model->num_layers - 1].neurons[i].output);
    }

    destroyMyAI(weather_model);
}

void image_compression() {
    printf("Image Compression (Simplified)\n");

    #define IMAGE_SIZE 64
    #define COMPRESSED_SIZE 16

    int layer_sizes[] = {IMAGE_SIZE * IMAGE_SIZE, COMPRESSED_SIZE * COMPRESSED_SIZE, IMAGE_SIZE * IMAGE_SIZE};
    myAI* autoencoder = createMyAI(layer_sizes, 3, 0.01, 32);

    // 训练自动编码器
    for (int epoch = 0; epoch < 10000; epoch++) {
        double total_error = 0.0;

        double input[IMAGE_SIZE * IMAGE_SIZE];
        
        // 生成随机的"图像"数据
        for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            input[i] = random_double(0, 1);
        }

        forward(autoencoder, input);
        backward(autoencoder, input, input);  // 自动编码器的目标是重构输入

        total_error += calculateError(autoencoder, input, IMAGE_SIZE * IMAGE_SIZE);

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Average Error: %f\n", epoch, total_error / (IMAGE_SIZE * IMAGE_SIZE));
        }
    }

    // 测试图像压缩
    printf("Testing image compression:\n");
    double test_image[IMAGE_SIZE * IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        test_image[i] = random_double(0, 1);
    }

    forward(autoencoder, test_image);

    printf("Original image size: %d pixels\n", IMAGE_SIZE * IMAGE_SIZE);
    printf("Compressed image size: %d pixels\n", COMPRESSED_SIZE * COMPRESSED_SIZE);

    double mse = 0.0;
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        double diff = test_image[i] - autoencoder->layers[autoencoder->num_layers - 1].neurons[i].output;
        mse += diff * diff;
    }
    mse /= (IMAGE_SIZE * IMAGE_SIZE);

    printf("Mean Squared Error: %f\n", mse);

    destroyMyAI(autoencoder);
}

double random_double(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

void generate_random_text(const char* language, myChar& text) {
    const char* words_english[] = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "I"};
    const char* words_spanish[] = {"el", "ser", "a", "de", "y", "un", "en", "que", "haber", "yo"};
    const char* words_french[] = {"le", "être", "à", "de", "et", "un", "en", "que", "avoir", "je"};
    const char* words_german[] = {"der", "sein", "zu", "von", "und", "ein", "in", "dass", "haben", "ich"};
    const char* words_italian[] = {"il", "essere", "a", "di", "e", "un", "in", "che", "avere", "io"};

    const char** words;
    int num_words;

    if (strcmp(language, "English") == 0) {
        words = words_english;
        num_words = sizeof(words_english) / sizeof(words_english[0]);
    } else if (strcmp(language, "Spanish") == 0) {
        words = words_spanish;
        num_words = sizeof(words_spanish) / sizeof(words_spanish[0]);
    } else if (strcmp(language, "French") == 0) {
        words = words_french;
        num_words = sizeof(words_french) / sizeof(words_french[0]);
    } else if (strcmp(language, "German") == 0) {
        words = words_german;
        num_words = sizeof(words_german) / sizeof(words_german[0]);
    } else if (strcmp(language, "Italian") == 0) {
        words = words_italian;
        num_words = sizeof(words_italian) / sizeof(words_italian[0]);
    } else {
        printf("Unsupported language: %s\n", language);
        return;
    }

    for (int i = 0; i < SAMPLE_LENGTH; i++) {
        int word_index = rand() % num_words;
        text.cat(words[word_index]);
        text.cat(" ");
    }
}