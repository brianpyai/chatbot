import random
import time

class Text:
    Black = lambda s: "\033[30m%s\033[0m" % s
    Red = lambda s: "\033[31m%s\033[0m" % s
    Green = lambda s: "\033[32m%s\033[0m" % s
    Yellow = lambda s: "\033[33m%s\033[0m" % s
    Blue = lambda s: "\033[34m%s\033[0m" % s
    Purple = lambda s: "\033[35m%s\033[0m" % s
    Cyan = lambda s: "\033[36m%s\033[0m" % s
    Gray = lambda s: "\033[37m%s\033[0m" % s
    DarkGray = lambda s: "\033[90m%s\033[0m" % s
    LightRed = lambda s: "\033[91m%s\033[0m" % s
    LightGreen = lambda s: "\033[92m%s\033[0m" % s
    LightYellow = lambda s: "\033[93m%s\033[0m" % s
    LightBlue = lambda s: "\033[94m%s\033[0m" % s
    LightPurple = lambda s: "\033[95m%s\033[0m" % s
    LightCyan = lambda s: "\033[96m%s\033[0m" % s
    White = lambda s: "\033[97m%s\033[0m" % s

playerX = 'X'
playerO = 'O'
currentPlayer = None
computerPlayer = None
humanPlayer = None
gameActive = True
board = [''] * 9
playerScore = 0
computerScore = 0

winningConditions = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

def drawBoard():
    print("\n" * 50)  # 清屏
    print(Text.White(f"玩家({humanPlayer}): {playerScore}  電腦({computerPlayer}): {computerScore}"))
    print(Text.White("-------------"))
    for i in range(3):
        print(Text.White("|"), end="")
        for j in range(3):
            index = i * 3 + j
            if board[index] == playerX:
                print(f" {Text.Red(board[index])} ", end="")
            elif board[index] == playerO:
                print(f" {Text.Blue(board[index])} ", end="")
            else:
                print(f" {Text.Gray(str(index + 1))} ", end="")
            print(Text.White("|"), end="")
        print("\n" + Text.White("-------------"))
    
    if currentPlayer == humanPlayer:
        print(Text.Green(f"您的回合，請輸入1-9來放置 {humanPlayer}"))
    else:
        print(Text.Yellow("電腦正在思考..."))

def initGame():
    global board, gameActive, currentPlayer, humanPlayer, computerPlayer
    board = [''] * 9
    gameActive = True
    
    if random.random() < 0.5:
        currentPlayer = playerX
        humanPlayer = playerX
        computerPlayer = playerO
    else:
        currentPlayer = playerO
        humanPlayer = playerO
        computerPlayer = playerX

    drawBoard()

    if currentPlayer == computerPlayer:
        time.sleep(0.5)
        compMove()

def makeMove(index, player):
    global gameActive, playerScore, computerScore, currentPlayer
    board[index] = player
    drawBoard()
    if checkWin(player):
        gameActive = False
        if player == humanPlayer:
            playerScore += 1
            print(Text.Green("恭喜你贏了！"))
        else:
            computerScore += 1
            print(Text.Red("電腦贏了！"))
        time.sleep(2)
        initGame()
        return
    if checkDraw():
        gameActive = False
        print(Text.Yellow("平局！"))
        time.sleep(2)
        initGame()
        return
    currentPlayer = computerPlayer if player == humanPlayer else humanPlayer

def compMove():
    if not gameActive:
        return
    
    bestMove = findBestMove()
    if bestMove != -1:
        makeMove(bestMove, computerPlayer)

def findBestMove():
    winMove = findWinningMove(computerPlayer)
    if winMove != -1:
        return winMove

    blockMove = findWinningMove(humanPlayer)
    if blockMove != -1:
        return blockMove

    if board[4] == '':
        return 4

    corners = [0, 2, 6, 8]
    availableCorners = [i for i in corners if board[i] == '']
    if availableCorners:
        return random.choice(availableCorners)

    availableCells = [i for i, cell in enumerate(board) if cell == '']
    return random.choice(availableCells) if availableCells else -1

def findWinningMove(player):
    for i in range(len(board)):
        if board[i] == '':
            board[i] = player
            if checkWin(player):
                board[i] = ''
                return i
            board[i] = ''
    return -1

def checkWin(player):
    return any(all(board[i] == player for i in condition) for condition in winningConditions)

def checkDraw():
    return all(cell != '' for cell in board)

def handleUserInput():
    global gameActive
    while gameActive and currentPlayer == humanPlayer:
        try:
            move = int(input("請輸入你的移動 (1-9): ")) - 1
            if 0 <= move <= 8 and board[move] == '':
                makeMove(move, humanPlayer)
                if gameActive:
                    time.sleep(0.5)
                    compMove()
            else:
                print(Text.Red("無效的移動，請重試。"))
        except ValueError:
            print(Text.Red("請輸入有效的數字。"))

if __name__ == "__main__":
    print(Text.Cyan("歡迎來到井字棋遊戲！"))
    initGame()
    while True:
        handleUserInput()
        if input(Text.Purple("按 Enter 開始新遊戲，或輸入 'q' 退出: ")).lower() == 'q':
            break
    print(Text.Cyan("感謝您的遊玩，再見！"))