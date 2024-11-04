// 需要包含的標頭檔和巨集定義
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

// 定義棋盤大小
#define BOARD_SIZE 9  
// 定義輸出層大小：9個策略輸出 + 1個價值輸出
#define OUTPUT_SIZE 10

// 激活函數及其導數
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

// 矩陣運算函數
double** allocate_matrix(int rows, int cols) {
    double **matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) {
        fprintf(stderr, "記憶體分配失敗\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < rows; i++) {
        matrix[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix[i]) {
            fprintf(stderr, "記憶體分配失敗\n");
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

void free_matrix(double **matrix, int rows) {
    if (!matrix) return;
    for(int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

// 神經網路結構
typedef struct {
    int num_layers;
    int *layer_sizes;
    double ***weights; // 權重矩陣
    double **biases;   // 偏差向量
    double **activations; // 激活值（輸出）
    double **zs;          // 加權輸入（未激活）
    double **deltas;      // 誤差項（反向傳播用）
} NeuralNetwork;

// 初始化神經網路
NeuralNetwork* init_network(int num_layers, int *layer_sizes) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "記憶體分配失敗\n");
        exit(EXIT_FAILURE);
    }
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    if (!nn->layer_sizes) {
        fprintf(stderr, "記憶體分配失敗\n");
        free(nn);
        exit(EXIT_FAILURE);
    }
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    nn->weights = (double***)malloc((num_layers - 1) * sizeof(double**));
    nn->biases = (double**)malloc((num_layers - 1) * sizeof(double*));
    nn->activations = (double**)malloc(num_layers * sizeof(double*));
    nn->zs = (double**)malloc((num_layers - 1) * sizeof(double*));
    nn->deltas = (double**)malloc((num_layers - 1) * sizeof(double*));

    if (!nn->weights || !nn->biases || !nn->activations || !nn->zs || !nn->deltas) {
        fprintf(stderr, "記憶體分配失敗\n");
        free(nn->layer_sizes);
        free(nn);
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < num_layers; i++) {
        nn->activations[i] = (double*)calloc(nn->layer_sizes[i], sizeof(double));
        if (!nn->activations[i]) {
            fprintf(stderr, "記憶體分配失敗\n");
            exit(EXIT_FAILURE);
        }
        if(i < num_layers -1) {
            nn->weights[i] = allocate_matrix(nn->layer_sizes[i+1], nn->layer_sizes[i]);
            nn->biases[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));
            nn->zs[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));
            nn->deltas[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));

            if (!nn->biases[i] || !nn->zs[i] || !nn->deltas[i]) {
                fprintf(stderr, "記憶體分配失敗\n");
                exit(EXIT_FAILURE);
            }

            // 隨機初始化權重和偏差 
            for(int j = 0; j < nn->layer_sizes[i+1]; j++) {
                nn->biases[i][j] = ((double)rand() / RAND_MAX) * 2 -1; // [-1,1]
                for(int k = 0; k < nn->layer_sizes[i]; k++)
                    nn->weights[i][j][k] = ((double)rand() / RAND_MAX) * 2 -1; // [-1,1]
            }
        }
    }

    return nn;
}

// 前向傳播
void forward(NeuralNetwork *nn, double *input) {
    // 輸入層激活值
    memcpy(nn->activations[0], input, nn->layer_sizes[0] * sizeof(double));

    // 從第一個隱藏層開始
    for(int l = 1; l < nn->num_layers; l++) {
        int rows = nn->layer_sizes[l];
        int cols = nn->layer_sizes[l-1];

        // 獲取權重和偏差
        double **W = nn->weights[l-1]; // 權重矩陣: rows x cols
        double *a_prev = nn->activations[l-1]; // 前一層激活值

        // 計算 z = W * a_prev + b
        for(int i = 0; i < rows; i++) {
            double z = 0.0;
            for(int j = 0; j < cols; j++) {
                z += W[i][j] * a_prev[j];
            }
            z += nn->biases[l-1][i];
            nn->zs[l-1][i] = z;
            nn->activations[l][i] = sigmoid(z);
        }
    }
}

// 反向傳播
void backward(NeuralNetwork *nn, double *target_policy, double target_value, double learning_rate) {
    int L = nn->num_layers -1;
    // 計算輸出層的 delta
    for(int i = 0; i < nn->layer_sizes[L]; i++) {
        double a = nn->activations[L][i];
        if(i < BOARD_SIZE) {
            // 策略輸出（前9個節點）
            nn->deltas[L-1][i] = (a - target_policy[i]) * sigmoid_derivative(a);
        } else {
            // 價值輸出（第10個節點）
            nn->deltas[L-1][i] = (a - target_value) * sigmoid_derivative(a);
        }
    }

    // 反向傳播誤差
    for(int l = L -1; l >= 1; l--) {
        int curr_size = nn->layer_sizes[l];
        int next_size = nn->layer_sizes[l+1];

        for(int i = 0; i < curr_size; i++) {
            double sum = 0.0;
            for(int j = 0; j < next_size; j++) {
                sum += nn->weights[l][j][i] * nn->deltas[l][j];
            }
            nn->deltas[l-1][i] = sum * sigmoid_derivative(nn->activations[l][i]);
        }
    }

    // 更新權重和偏差
    for(int l = 0; l < L; l++) {
        int rows = nn->layer_sizes[l+1];
        int cols = nn->layer_sizes[l];

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                nn->weights[l][i][j] -= learning_rate * nn->deltas[l][i] * nn->activations[l][j];
            }
            nn->biases[l][i] -= learning_rate * nn->deltas[l][i];
        }
    }
}

// 儲存神經網路到檔案
void save_network(NeuralNetwork *nn, const char *filename) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) {
        printf("無法開啟檔案儲存網路。\n");
        return;
    }

    // 儲存層數和每層大小
    fwrite(&nn->num_layers, sizeof(int), 1, fp);
    fwrite(nn->layer_sizes, sizeof(int), nn->num_layers, fp);

    // 儲存權重和偏差
    for(int l = 0; l < nn->num_layers -1; l++) {
        int rows = nn->layer_sizes[l+1];
        int cols = nn->layer_sizes[l];
        for(int i = 0; i < rows; i++) {
            fwrite(nn->weights[l][i], sizeof(double), cols, fp);
        }
        fwrite(nn->biases[l], sizeof(double), rows, fp);
    }

    fclose(fp);
}

// 從檔案載入神經網路
NeuralNetwork* load_network(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        printf("無法開啟檔案載入網路。\n");
        return NULL;
    }

    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!nn) {
        fprintf(stderr, "記憶體分配失敗\n");
        exit(EXIT_FAILURE);
    }
    fread(&nn->num_layers, sizeof(int), 1, fp);
    nn->layer_sizes = (int*)malloc(nn->num_layers * sizeof(int));
    if (!nn->layer_sizes) {
        fprintf(stderr, "記憶體分配失敗\n");
        free(nn);
        exit(EXIT_FAILURE);
    }
    fread(nn->layer_sizes, sizeof(int), nn->num_layers, fp);

    nn->weights = (double***)malloc((nn->num_layers - 1) * sizeof(double**));
    nn->biases = (double**)malloc((nn->num_layers - 1) * sizeof(double*));
    nn->activations = (double**)malloc(nn->num_layers * sizeof(double*));
    nn->zs = (double**)malloc((nn->num_layers - 1) * sizeof(double*));
    nn->deltas = (double**)malloc((nn->num_layers - 1) * sizeof(double*));

    if (!nn->weights || !nn->biases || !nn->activations || !nn->zs || !nn->deltas) {
        fprintf(stderr, "記憶體分配失敗\n");
        free(nn->layer_sizes);
        free(nn);
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < nn->num_layers; i++) {
        nn->activations[i] = (double*)calloc(nn->layer_sizes[i], sizeof(double));
        if (!nn->activations[i]) {
            fprintf(stderr, "記憶體分配失敗\n");
            exit(EXIT_FAILURE);
        }
        if(i < nn->num_layers -1) {
            nn->weights[i] = allocate_matrix(nn->layer_sizes[i+1], nn->layer_sizes[i]);
            nn->biases[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));
            nn->zs[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));
            nn->deltas[i] = (double*)calloc(nn->layer_sizes[i+1], sizeof(double));

            if (!nn->biases[i] || !nn->zs[i] || !nn->deltas[i]) {
                fprintf(stderr, "記憶體分配失敗\n");
                exit(EXIT_FAILURE);
            }

            // 載入權重和偏差
            int rows = nn->layer_sizes[i+1];
            int cols = nn->layer_sizes[i];
            for(int j = 0; j < rows; j++) {
                fread(nn->weights[i][j], sizeof(double), cols, fp);
            }
            fread(nn->biases[i], sizeof(double), rows, fp);
        }
    }

    fclose(fp);
    return nn;
}

// 釋放神經網路
void free_network(NeuralNetwork *nn) {
    if (!nn) return;
    for(int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        if(i < nn->num_layers -1) {
            free_matrix(nn->weights[i], nn->layer_sizes[i+1]);
            free(nn->biases[i]);
            free(nn->zs[i]);
            free(nn->deltas[i]);
        }
    }
    free(nn->activations);
    free(nn->weights);
    free(nn->biases);
    free(nn->zs);
    free(nn->deltas);
    free(nn->layer_sizes);
    free(nn);
}

// 檢測遊戲是否結束
int check_winner(int *board) {
    int win_conditions[8][3] = {
        {0,1,2},{3,4,5},{6,7,8}, // 行
        {0,3,6},{1,4,7},{2,5,8}, // 列
        {0,4,8},{2,4,6}          // 對角線
    };

    // 檢查是否有勝利者
    for(int i=0; i<8; i++) {
        int a = win_conditions[i][0];
        int b = win_conditions[i][1];
        int c = win_conditions[i][2];
        if(board[a] != 0 && board[a] == board[b] && board[b] == board[c])
            return board[a]; // 返回1或-1，表示哪個玩家勝利
    }

    // 檢查是否平局
    int draw = 1;
    for(int i=0; i<BOARD_SIZE; i++) {
        if(board[i] == 0) {
            draw = 0;
            break;
        }
    }
    if(draw) return 0; // 平局

    // 遊戲未結束
    return 2;
}

// 獲取合法的下棋位置
void get_valid_moves(int *board, int *valid_moves, int *num_moves) {
    *num_moves = 0;
    for(int i=0; i<BOARD_SIZE; i++) {
        if(board[i] == 0) {
            valid_moves[*num_moves] = i;
            (*num_moves)++;
        }
    }
}

// MCTS的節點結構（保持不變）
typedef struct MCTSNode {
    int state[BOARD_SIZE];     // 棋盤狀態
    int player;                // 當前玩家
    int move;                  // 導致此節點的移動
    struct MCTSNode *parent;   // 父節點
    struct MCTSNode **children;// 子節點列表
    int num_children;          // 子節點數量
    int wins;                  // 勝利次數
    int visits;                // 訪問次數
} MCTSNode;

// MCTS相關函數（保持不變）

// 創建新節點
MCTSNode* create_node(int *state, int player, int move, MCTSNode *parent) {
    MCTSNode *node = (MCTSNode*)malloc(sizeof(MCTSNode));
    memcpy(node->state, state, sizeof(int) * BOARD_SIZE);
    node->player = player;
    node->move = move;
    node->parent = parent;
    node->children = NULL;
    node->num_children = 0;
    node->wins = 0;
    node->visits = 0;
    return node;
}

// 釋放節點
void free_mcts_node(MCTSNode *node) {
    if(node->children != NULL) {
        for(int i = 0; i < node->num_children; i++) {
            free_mcts_node(node->children[i]);
        }
        free(node->children);
    }
    free(node);
}

// MCTS的選擇階段（UCB1）
MCTSNode* mcts_selection(MCTSNode *node) {
    while(node->num_children > 0) {
        double best_ucb1 = -1.0;
        MCTSNode *best_child = NULL;
        for(int i = 0; i < node->num_children; i++) {
            MCTSNode *child = node->children[i];
            double ucb1 = (double)child->wins / (child->visits + 1e-6) +
                          sqrt(2 * log(node->visits + 1) / (child->visits + 1e-6));
            if(ucb1 > best_ucb1) {
                best_ucb1 = ucb1;
                best_child = child;
            }
        }
        node = best_child;
    }
    return node;
}

// 展開節點
void mcts_expansion(MCTSNode *node) {
    int valid_moves[BOARD_SIZE];
    int num_moves;
    get_valid_moves(node->state, valid_moves, &num_moves);

    node->children = (MCTSNode**)malloc(sizeof(MCTSNode*) * num_moves);
    node->num_children = num_moves;

    for(int i = 0; i < num_moves; i++) {
        int new_state[BOARD_SIZE];
        memcpy(new_state, node->state, sizeof(int) * BOARD_SIZE);
        new_state[valid_moves[i]] = node->player;
        node->children[i] = create_node(new_state, -node->player, valid_moves[i], node);
    }
}

// 模擬隨機遊戲
int mcts_simulation(MCTSNode *node) {
    int current_player = node->player;
    int state[BOARD_SIZE];
    memcpy(state, node->state, sizeof(int) * BOARD_SIZE);

    int result = check_winner(state);
    while(result == 2) {
        int valid_moves[BOARD_SIZE];
        int num_moves;
        get_valid_moves(state, valid_moves, &num_moves);

        int move = valid_moves[rand() % num_moves];
        state[move] = current_player;

        result = check_winner(state);
        current_player = -current_player;
    }
    return result;
}

// 反向傳播
void mcts_backpropagation(MCTSNode *node, int result) {
    while(node != NULL) {
        node->visits++;
        if(node->player == result) {
            node->wins++;
        }
        node = node->parent;
    }
}

// MCTS主函數
int mcts(int *board, int player, int simulations) {
    MCTSNode *root = create_node(board, player, -1, NULL);

    for(int i = 0; i < simulations; i++) {
        // 選擇
        MCTSNode *node = mcts_selection(root);

        // 展開
        int result = check_winner(node->state);
        if(result == 2 && node->visits > 0) {
            mcts_expansion(node);
            if(node->num_children > 0)
                node = node->children[rand() % node->num_children];
        }

        // 模擬
        result = mcts_simulation(node);

        // 反向傳播
        mcts_backpropagation(node, result);
    }

    // 選擇訪問次數最多的子節點
    int best_move = -1;
    int max_visits = -1;
    for(int i = 0; i < root->num_children; i++) {
        MCTSNode *child = root->children[i];
        if(child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move;
        }
    }

    free_mcts_node(root);
    return best_move;
}

// 使用MCTS的對手移動
int opponent_move(int *board, int opponent_mark) {
    return mcts(board, opponent_mark, 1000); // 進行1000次模擬
}

// 改進的訓練函數
void self_play_training(NeuralNetwork *nn, int epochs, double learning_rate) {
    char continue_training = 'y';
    while(continue_training == 'y' || continue_training == 'Y') {
        for(int epoch = 0; epoch < epochs; epoch++) {
            int board[BOARD_SIZE] = {0};
            int game_over = 0;
            int moves_record[BOARD_SIZE];
            int move_count = 0;

            // 記錄每一步的棋盤狀態
            int board_states[BOARD_SIZE][BOARD_SIZE];

            // 隨機決定先後手
            int current_player = rand() % 2 == 0 ? 1 : -1;

            while(!game_over) {
                // 獲取合法移動
                int valid_moves[BOARD_SIZE];
                int num_moves;
                get_valid_moves(board, valid_moves, &num_moves);

                // 記錄當前棋盤狀態
                memcpy(board_states[move_count], board, sizeof(int) * BOARD_SIZE);

                if(current_player == 1) {
                    // AI的回合
                    // 準備神經網路的輸入
                    double input[BOARD_SIZE];
                    for(int i=0; i<BOARD_SIZE; i++) {
                        input[i] = board[i] * current_player; // 視角轉換
                    }

                    // 前向傳播
                    forward(nn, input);

                    // 獲取輸出層
                    double *output = nn->activations[nn->num_layers - 1];

                    // 策略輸出為前9個節點
                    // 選擇評分最高的合法移動
                    int best_move = -1;
                    double best_value = -INFINITY;
                    for(int i=0; i<num_moves; i++) {
                        int idx = valid_moves[i];
                        if(output[idx] > best_value) {
                            best_value = output[idx];
                            best_move = idx;
                        }
                    }

                    // 如果沒有找到合法的移動（理論上不應該發生）
                    if(best_move == -1) {
                        best_move = valid_moves[rand() % num_moves];
                    }

                    // 執行移動
                    board[best_move] = current_player;
                    moves_record[move_count++] = best_move;
                } else {
                    // 對手的回合，使用MCTS
                    int opp_move = opponent_move(board, current_player);
                    board[opp_move] = current_player;
                }

                // 檢查遊戲是否結束
                int result = check_winner(board);
                if(result != 2) {
                    game_over = 1;

                    // 回合結束，基於遊戲結果進行訓練
                    // result: 1=勝，0=平局，-1=負

                    for(int i=0; i<move_count; i++) {
                        // 準備輸入
                        double train_input[BOARD_SIZE];
                        for(int j=0; j<BOARD_SIZE; j++) {
                            train_input[j] = board_states[i][j];
                        }

                        // 準備目標輸出
                        double target_output[OUTPUT_SIZE] = {0};
                        // 策略目標
                        int move = moves_record[i];
                        target_output[move] = 1.0; // 將實際走的那一步設為1，其餘為0

                        // 價值目標
                        double target_value;
                        if(result == 1) target_value = 1.0;
                        else if(result == 0) target_value = 0.5;
                        else target_value = 0.0;

                        // 前向傳播和反向傳播
                        forward(nn, train_input);
                        backward(nn, target_output, target_value, learning_rate);
                    }
                } else {
                    // 切換玩家
                    current_player *= -1;
                }
            }

            if(epoch % 1000 == 0)
                printf("Epoch %d completed.\n", epoch);
        }

        // 詢問是否繼續訓練
        printf("是否繼續訓練？（y/n）：");
        scanf(" %c", &continue_training);
    }
}

// 顯示棋盤
void display_board(int *board) {
    char symbols[] = {' ', 'X', 'O'};
    printf("\n");
    for(int i=0; i<BOARD_SIZE; i++) {
        int val = board[i];
        char c = symbols[val == 0 ? 0 : (val == 1 ? 1 : 2)];
        printf(" %c ", c);
        if((i+1) % 3 == 0 && i != BOARD_SIZE -1)
            printf("\n---+---+---\n");
        else if(i != BOARD_SIZE -1)
            printf("|");
    }
    printf("\n");
}

// 定義遊戲記錄結構
typedef struct {
    int board_states[BOARD_SIZE][BOARD_SIZE]; // 每一步的棋盤狀態
    int moves[BOARD_SIZE]; // 每一步的移動
    int move_count; // 移動總數
    int result; // 結果（1=勝，0=平局，-1=負）
} GameRecord;

// 最大遊戲記錄數
#define MAX_GAME_RECORDS 1000

// 與用戶對戰
void user_vs_ai(NeuralNetwork *nn, GameRecord *game_records, int *num_game_records) {
    int board[BOARD_SIZE] = {0};
    int game_over = 0;
    GameRecord current_game;
    current_game.move_count = 0;

    int move_count = 0;
    int last_move = -1;

    // 隨機決定先後手
    int current_player = rand() % 2 == 0 ? 1 : -1;

    if(current_player == 1) {
        printf("您先手（X）\n");
    } else {
        printf("AI先手（O）\n");
    }

    while(!game_over) {
        display_board(board);

        if(current_player == 1) {
            // 用戶輸入
            int move;
            printf("請輸入你要下的位置（1-9）：");
            scanf("%d", &move);
            move--; // 轉換為0索引
            if(move < 0 || move >= BOARD_SIZE || board[move] != 0) {
                printf("無效的移動，請重試。\n");
                continue;
            }
            board[move] = current_player;
            last_move = move;
        } else {
            // AI下棋
            // 獲取合法移動
            int valid_moves[BOARD_SIZE];
            int num_moves;
            get_valid_moves(board, valid_moves, &num_moves);

            // 準備輸入
            double input[BOARD_SIZE];
            for(int i=0; i<BOARD_SIZE; i++) {
                input[i] = board[i] * current_player;
            }

            // 前向傳播
            forward(nn, input);

            // 獲取輸出層
            double *output = nn->activations[nn->num_layers - 1];

            // 策略輸出為前9個節點，價值輸出為第10個節點
            // 我們可以結合策略和價值來選擇行動

            // 計算每個合法移動的總評分 = 策略評分 + 預期價值
            int best_move = -1;
            double best_total_value = -INFINITY;
            for(int i=0; i<num_moves; i++) {
                int idx = valid_moves[i];
                double move_value = output[idx]; // 策略評分
                // 模擬執行該移動，並預測新狀態的價值
                int temp_board[BOARD_SIZE];
                memcpy(temp_board, board, sizeof(int) * BOARD_SIZE);
                temp_board[idx] = current_player;
                double next_input[BOARD_SIZE];
                for(int j=0; j<BOARD_SIZE; j++) {
                    next_input[j] = temp_board[j] * (-current_player); // 對手視角
                }
                forward(nn, next_input);
                double next_value = nn->activations[nn->num_layers -1][OUTPUT_SIZE -1]; // 價值輸出

                double total_value = move_value + next_value; // 結合策略和價值
                if(total_value > best_total_value) {
                    best_total_value = total_value;
                    best_move = idx;
                }
            }

            // 如果沒有找到合法的移動（理論上不應該發生）
            if(best_move == -1) {
                best_move = valid_moves[rand() % num_moves];
            }

            printf("AI選擇了位置 %d\n", best_move + 1);
            board[best_move] = current_player;
            last_move = best_move;
        }

        // 記錄棋盤狀態和移動

        memcpy(current_game.board_states[move_count], board, sizeof(int) * BOARD_SIZE);
        current_game.moves[move_count] = last_move;
        move_count++;
        current_game.move_count = move_count;

        // 檢查遊戲是否結束
        int result = check_winner(board);
        if(result != 2) {
            display_board(board);
            if(result == 0) {
                printf("遊戲平局！\n");
            } else if(result == 1) {
                printf("X贏了！\n");
            } else {
                printf("O赢了！\n");
            }
            game_over = 1;

            // 記錄遊戲結果
            current_game.result = result;
            // 添加到遊戲記錄
            if(*num_game_records < MAX_GAME_RECORDS) {
                game_records[*num_game_records] = current_game;
                (*num_game_records)++;
            } else {
                printf("遊戲記錄已滿，無法記錄更多遊戲。\n");
            }
        } else {
            current_player *= -1;
        }
    }
}

// 從遊戲記錄訓練模型
void train_from_game_records(NeuralNetwork *nn, GameRecord *game_records, int num_game_records, double learning_rate) {
    for(int g=0; g<num_game_records; g++) {
        GameRecord *game = &game_records[g];
        int result = game->result; // 最終結果

        for(int i=0; i<game->move_count; i++) {
            // 準備輸入
            double train_input[BOARD_SIZE];
            for(int j=0; j<BOARD_SIZE; j++) {
                train_input[j] = game->board_states[i][j];
            }

            // 準備目標輸出
            double target_output[OUTPUT_SIZE] = {0};
            // 策略目標
            int move = game->moves[i];
            target_output[move] = 1.0; // 將實際走的那一步設為1，其餘為0

            // 價值目標
            double target_value;
            if(result == 1) target_value = 1.0;
            else if(result == 0) target_value = 0.5;
            else target_value = 0.0;

            // 前向傳播和反向傳播
            forward(nn, train_input);
            backward(nn, target_output, target_value, learning_rate);
        }
    }
}

// 主函數
int main() {
    srand(time(NULL));

    // 定義網路結構
    int layers[] = {9, 256, 256, OUTPUT_SIZE};
    int num_layers = sizeof(layers) / sizeof(layers[0]);

    // 檢查模型檔是否存在
    struct stat buffer;
    int model_exists = (stat("tic_tac_toe_nn.dat", &buffer) == 0);

    NeuralNetwork *nn;

    if(model_exists) {
        // 如果模型檔存在，載入模型
        printf("檢測到已訓練嘅模型，加載模型緊...\n");
        nn = load_network("tic_tac_toe_nn.dat");
    } else {
        // 如果模型檔不存在，初始化新的神經網路
        printf("未檢測到已訓練的模型，正在初始化新的神經網路...\n");
        nn = init_network(num_layers, layers);

        // 訓練參數
        int training_epochs = 100000; // 可以根據需要調整訓練輪數
        double learning_rate = 0.001;

        // 進行自我對戰訓練
        printf("開始自我對戰訓練...\n");
        self_play_training(nn, training_epochs, learning_rate);
        printf("訓練完成！\n");

        // 保存网络
        save_network(nn, "tic_tac_toe_nn.dat");
        printf("模型已保存到 tic_tac_toe_nn.dat\n");
    }

    // 準備遊戲記錄

    GameRecord game_records[MAX_GAME_RECORDS];
    int num_game_records = 0;

    // 與用戶對戰
    printf("而家可以與AI對戰了！\n");
    char continue_playing = 'y';
    while(continue_playing == 'y' || continue_playing == 'Y') {
        user_vs_ai(nn, game_records, &num_game_records);

        // 問用戶是否繼續遊戲
        printf("是否繼續遊戲？（y/n）：");
        scanf(" %c", &continue_playing);
    }

    // 使用遊戲記錄進行訓練
    if(num_game_records > 0) {
        printf("正在使用對戰記錄訓練模型...\n");
        double learning_rate = 0.3;
        train_from_game_records(nn, game_records, num_game_records, learning_rate);
        printf("訓練完成！\n");

        // 保存網絡
        save_network(nn, "tic_tac_toe_nn.dat");
        printf("模型已保存到文件 tic_tac_toe_nn.dat\n");
    }

    // 問用戶是否繼續訓練
    char continue_training;
    printf("是否繼續訓練？（y/n）：");
    scanf(" %c", &continue_training);
    if(continue_training == 'y' || continue_training == 'Y') {
        // 訓練參數
        int training_epochs = 100000; // 可以根據需要調整訓練輪數
        double learning_rate = 0.001;

        // 進行自我對戰訓練
        printf("开始自我对战训练...\n");
        self_play_training(nn, training_epochs, learning_rate);
        printf("訓練完成！\n");

        // 保存網絡
        save_network(nn, "tic_tac_toe_nn.dat");
        printf("模型已保存到文件 tic_tac_toe_nn.dat\n");
    }

    // 釋放網絡
    free_network(nn);

    return 0;
}
