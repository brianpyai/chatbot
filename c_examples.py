import gradio as gr
import subprocess
import os

# 創建臨時目錄
TEMP_BIN_DIR = os.path.expanduser("~/tempbin")
os.makedirs(TEMP_BIN_DIR, exist_ok=True)

EXAMPLE_CODES = [
    ("Bubblesort", """
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void bubble_sort(int array[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
            }
        }
    }
}

int main() {
    int array[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(array) / sizeof(array[0]);
    
    bubble_sort(array, n);
    
    printf("Sorted array:\\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),
    ("Binary Search", """
#include <stdio.h>

int binary_search(int array[], int n, int key) {
    int left = 0;
    int right = n - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        if (array[mid] == key) {
            return mid;
        } else if (array[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

int main() {
    int array[] = {2, 4, 6, 8, 10};
    int n = sizeof(array) / sizeof(array[0]);
    
    int key = 6;
    int result = binary_search(array, n, key);
    
    if (result != -1) {
        printf("Element found at index %d\\n", result);
    } else {
        printf("Element not found in array\\n");
    }
    return 0;
}
    """),
    ("Reverse String", """
#include <stdio.h>
#include <string.h>

void reverse_string(char str[]) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

int main() {
    char str[] = "Hello, world!";
    
    reverse_string(str);
    
    printf("Reversed string: %s\\n", str);
    return 0;
}
    """),
    ("Calculate Power", """
#include <stdio.h>

int power(int base, int exponent) {
    int result = 1;
    
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    
    return result;
}

int main() {
    int base = 2;
    int exponent = 3;
    int result = power(base, exponent);
    
    printf("%d raised to the power of %d is %d\\n", base, exponent, result);
    return 0;
}
    """),
    ("Greatest Common Divisor", """
#include <stdio.h>

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main() {
    int num1 = 54;
    int num2 = 24;
    int result = gcd(num1, num2);
    
    printf("GCD of %d and %d is %d\\n", num1, num2, result);
    return 0;
}
    """),
    ("Least Common Multiple", """
#include <stdio.h>

int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main() {
    int num1 = 12;
    int num2 = 18;
    int result = lcm(num1, num2);
    
    printf("LCM of %d and %d is %d\\n", num1, num2, result);
    return 0;
}
    """),
    ("String Palindrome Check", """
#include <stdio.h>
#include <string.h>

int is_palindrome(char str[]) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        if (str[i] != str[n - i - 1]) {
            return 0;
        }
    }
    return 1;
}

int main() {
    char str[] = "racecar";
    
    if (is_palindrome(str)) {
        printf("%s is a palindrome.\\n", str);
    } else {
        printf("%s is not a palindrome.\\n", str);
    }
    return 0;
}
    """),
    ("Remove Duplicates from Sorted Array", """
#include <stdio.h>

int remove_duplicates(int array[], int n) {
    if (n == 0 || n == 1) {
        return n;
    }
    
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (array[i] != array[i + 1]) {
            array[j++] = array[i];
        }
    }
    
    array[j++] = array[n - 1];
    return j;
}

int main() {
    int array[] = {1, 1, 2, 2, 3, 4, 4, 5, 5};
    int n = sizeof(array) / sizeof(array[0]);
    
    int new_length = remove_duplicates(array, n);
    
    printf("Array after removing duplicates:\\n");
    for (int i = 0; i < new_length; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),
    ("Insertion Sort", """
#include <stdio.h>

void insertion_sort(int array[], int n) {
    for (int i = 1; i < n; i++) {
        int key = array[i];
        int j = i - 1;
        
        while (j >= 0 && array[j] > key) {
            array[j + 1] = array[j];
            j--;
        }
        
        array[j + 1] = key;
    }
}

int main() {
    int array[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(array) / sizeof(array[0]);
    
    insertion_sort(array, n);
    
    printf("Sorted array:\\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),
    
    ("Bubblesort", """
#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void bubble_sort(int array[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (array[j] > array[j + 1]) {
                swap(&array[j], &array[j + 1]);
            }
        }
    }
}

int main() {
    int array[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(array) / sizeof(array[0]);
    
    bubble_sort(array, n);
    
    printf("Sorted array:\\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),
    ("Binary Search", """
#include <stdio.h>

int binary_search(int array[], int n, int key) {
    int left = 0;
    int right = n - 1;
    
    while (left <= right) {
        int mid = (left + right) / 2;
        
        if (array[mid] == key) {
            return mid;
        } else if (array[mid] < key) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

int main() {
    int array[] = {2, 4, 6, 8, 10};
    int n = sizeof(array) / sizeof(array[0]);
    
    int key = 6;
    int result = binary_search(array, n, key);
    
    if (result != -1) {
        printf("Element found at index %d\\n", result);
    } else {
        printf("Element not found in array\\n");
    }
    return 0;
}
    """),
    ("Reverse String", """
#include <stdio.h>
#include <string.h>

void reverse_string(char str[]) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

int main() {
    char str[] = "Hello, world!";
    
    reverse_string(str);
    
    printf("Reversed string: %s\\n", str);
    return 0;
}
    """),
    ("Calculate Power", """
#include <stdio.h>

int power(int base, int exponent) {
    int result = 1;
    
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    
    return result;
}

int main() {
    int base = 2;
    int exponent = 3;
    int result = power(base, exponent);
    
    printf("%d raised to the power of %d is %d\\n", base, exponent, result);
    return 0;
}
    """),
    ("Greatest Common Divisor", """
#include <stdio.h>

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main() {
    int num1 = 54;
    int num2 = 24;
    int result = gcd(num1, num2);
    
    printf("GCD of %d and %d is %d\\n", num1, num2, result);
    return 0;
}
    """),
    ("Least Common Multiple", """
#include <stdio.h>

int lcm(int a, int b) {
    return (a / gcd(a, b)) * b;
}

int gcd(int a, int b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

int main() {
    int num1 = 12;
    int num2 = 18;
    int result = lcm(num1, num2);
    
    printf("LCM of %d and %d is %d\\n", num1, num2, result);
    return 0;
}
    """),
    ("String Palindrome Check", """
#include <stdio.h>
#include <string.h>

int is_palindrome(char str[]) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        if (str[i] != str[n - i - 1]) {
            return 0;
        }
    }
    return 1;
}

int main() {
    char str[] = "racecar";
    
    if (is_palindrome(str)) {
        printf("%s is a palindrome.\\n", str);
    } else {
        printf("%s is not a palindrome.\\n", str);
    }
    return 0;
}
    """),
    ("Remove Duplicates from Sorted Array", """
#include <stdio.h>

int remove_duplicates(int array[], int n) {
    if (n == 0 || n == 1) {
        return n;
    }
    
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (array[i] != array[i + 1]) {
            array[j++] = array[i];
        }
    }
    
    array[j++] = array[n - 1];
    return j;
}

int main() {
    int array[] = {1, 1, 2, 2, 3, 4, 4, 5, 5};
    int n = sizeof(array) / sizeof(array[0]);
    
    int new_length = remove_duplicates(array, n);
    
    printf("Array after removing duplicates:\\n");
    for (int i = 0; i < new_length; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),
    ("Insertion Sort", """
#include <stdio.h>

void insertion_sort(int array[], int n) {
    for (int i = 1; i < n; i++) {
        int key = array[i];
        int j = i - 1;
        
        while (j >= 0 && array[j] > key) {
            array[j + 1] = array[j];
            j--;
        }
        
        array[j + 1] = key;
    }
}

int main() {
    int array[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(array) / sizeof(array[0]);
    
    insertion_sort(array, n);
    
    printf("Sorted array:\\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\\n");
    return 0;
}
    """),


    ("HTML Table", """
#include <stdio.h>

int main() {
    printf("<table border='1'>\\n");
    printf("   <tr>\\n");
    printf("       <th>Name</th>\\n");
    printf("       <th>Age</th>\\n");
    printf("   </tr>\\n");
    printf("   <tr>\\n");
    printf("       <td>John</td>\\n");
    printf("       <td>25</td>\\n");
    printf("   </tr>\\n");
    printf("   <tr>\\n");
    printf("       <td>Jane</td>\\n");
    printf("       <td>30</td>\\n");
    printf("   </tr>\\n");
    printf("</table>\\n");
    return 0;
}
    """),
    ("HTML List", """
#include <stdio.h>

int main() {
    printf("<ul>\\n");
    printf("   <li>Apple</li>\\n");
    printf("   <li>Banana</li>\\n");
    printf("   <li>Cherry</li>\\n");
    printf("</ul>\\n");
    return 0;
}
    """),
    
]
EXAMPLE_CODES += [
    ("HTML Div", """
#include <stdio.h>

int main() {
    printf("<div style='background-color: lightblue; padding: 20px;'>\\n");
    printf("   This is a div element.\\n");
    printf("</div>\\n");
    return 0;
}
    """),
    ("HTML Span", """
#include <stdio.h>

int main() {
    printf("This is <span style='color: red;'>red</span> text.\\n");
    return 0;
}
    """),

    ("HTML Ordered List", """
#include <stdio.h>

int main() {
    printf("<ol>\\n");
    printf("   <li>First item</li>\\n");
    printf("   <li>Second item</li>\\n");
    printf("   <li>Third item</li>\\n");
    printf("</ol>\\n");
    return 0;
}
    """),
]
EXAMPLE_CODES += [
  
    ("HTML Nested List", """
#include <stdio.h>

int main() {
    printf("<ul>\\n");
    printf("   <li>Parent Item 1\\n");
    printf("       <ul>\\n");
    printf("           <li>Child Item 1</li>\\n");
    printf("           <li>Child Item 2</li>\\n");
    printf("       </ul>\\n");
    printf("   </li>\\n");
    printf("   <li>Parent Item 2</li>\\n");
    printf("</ul>\\n");
    return 0;
}
    """),

    ("HTML Definition List", """
#include <stdio.h>

int main() {
    printf("<dl>\\n");
    printf("   <dt>Term 1</dt>\\n");
    printf("   <dd>Term 1 description.</dd>\\n");
    printf("   <dt>Term 2</dt>\\n");
    printf("   <dd>Term 2 description.</dd>\\n");
    printf("</dl>\\n");
    return 0;
}
    """),

    ("HTML Preformatted Text", """
#include <stdio.h>

int main() {
    printf("<pre>\\n");
    printf("int main() {\\n");
    printf("    printf(\"Hello, world!\\n\");\\n");
    printf("    return 0;\\n");
    printf("}\\n");
    printf("</pre>\\n");
    return 0;
}
    """),

    ("HTML Anchors", """
#include <stdio.h>

int main() {
    printf("<a href='https://www.example.com'>Example Website</a>\\n");
    return 0;
}
    """),

]
EXAMPLE_CODES += [
    ("HTML Background Color", """
#include <stdio.h>

int main() {
    printf("<p style='background-color: lightblue;'>This text has a light blue background.</p>\\n");
    return 0;
}
    """),
    ("HTML Text Color", """
#include <stdio.h>

int main() {
    printf("<p style='color: red;'>This text is red.</p>\\n");
    return 0;
}
    """),
    ("HTML Font Size", """
#include <stdio.h>

int main() {
    printf("<p style='font-size: 24px;'>This text has a large font size.</p>\\n");
    return 0;
}
    """),
    ("HTML Font Family", """
#include <stdio.h>

int main() {
    printf("<p style='font-family: Arial, sans-serif;'>This text uses Arial font.</p>\\n");
    return 0;
}
    """),
    ("HTML Text Alignment", """
#include <stdio.h>

int main() {
    printf("<p style='text-align: center;'>This text is centered.</p>\\n");
    return 0;
}
    """),
    ("HTML Text Decoration", """
#include <stdio.h>

int main() {
    printf("<p style='text-decoration: underline;'>This text is underlined.</p>\\n");
    return 0;
}
    """),
    ("HTML Text Transform", """
#include <stdio.h>

int main() {
    printf("<p style='text-transform: uppercase;'>This text is in UPPERCASE.</p>\\n");
    return 0;
}
    """),
    ("HTML Text Shadow", """
#include <stdio.h>

int main() {
    printf("<p style='text-shadow: 2px 2px 2px black;'>This text has a shadow.</p>\\n");
    return 0;
}
    """)]
EXAMPLE_CODES += [
    ("HTML Div with Border", """
#include <stdio.h>

int main() {
    printf("<div style='border: 1px solid black; padding: 20px;'>\\n");
    printf("   This text is inside a div with a border.\\n");
    printf("</div>\\n");
    return 0;
}
    """),
    ("HTML Span with Style", """
#include <stdio.h>

int main() {
    printf("<span style='color: red; font-weight: bold;'>This is a red and bold span.</span>\\n");
    return 0;
}
    """),
    ("HTML Unordered List", """
#include <stdio.h>

int main() {
    printf("<ul>\\n");
    printf("   <li>Apple</li>\\n");
    printf("   <li>Banana</li>\\n");
    printf("   <li>Cherry</li>\\n");
    printf("</ul>\\n");
    return 0;
}
    """),
    ("HTML Ordered List", """
#include <stdio.h>

int main() {
    printf("<ol>\\n");
    printf("   <li>First item</li>\\n");
    printf("   <li>Second item</li>\\n");
    printf("   <li>Third item</li>\\n");
    printf("</ol>\\n");
    return 0;
}
    """),
    ("HTML Definition List", """
#include <stdio.h>

int main() {
    printf("<dl>\\n");
    printf("   <dt>Term</dt>\\n");
    printf("   <dd>Definition</dd>\\n");
    printf("</dl>\\n");
    return 0;
}
    """),
    ("HTML Description List", """
#include <stdio.h>

int main() {
    printf("<dl>\\n");
    printf("   <dt>Term 1</dt>\\n");
    printf("   <dd>Description 1</dd>\\n");
    printf("   <dt>Term 2</dt>\\n");
    printf("   <dd>Description 2</dd>\\n");
    printf("</dl>\\n");
    return 0;
}
    """),
    ("HTML Inline Element", """
#include <stdio.h>

int main() {
    printf("<p>This is a paragraph with an <mark>inline</mark> element.</p>\\n");
    return 0;
}
    """),
    ("HTML Block Element", """
#include <stdio.h>

int main() {
    printf("<p>This is a paragraph.</p>\\n");
    printf("<div>This is a div block element.</div>\\n");
    return 0;
}
    """),
    ("HTML Image", """
#include <stdio.h>

int main() {
    printf("<img src='http://127.0.0.1:9999/myimages/None_0793_2_-7944049407062523243.jpg' alt='Example Image'>");
    return 0;
}
    """),
]

EXAMPLE_CODES.reverse()
def compile_and_run(c_code, example_code):
    """
    將 C 程式碼編譯並執行,返回輸出結果
    """
    # 如果選擇了範例程式碼,則使用範例程式碼
    if example_code:
        c_code = example_code

    temp_c_file = os.path.join(TEMP_BIN_DIR, "temp.c")
    temp_bin_file = os.path.join(TEMP_BIN_DIR, "temp")

    with open(temp_c_file, "w") as f:
        f.write(c_code)

    try:
        subprocess.run(["g++", "-o", temp_bin_file, temp_c_file], cwd=TEMP_BIN_DIR, check=True)
    except subprocess.CalledProcessError as e:
        return f"g++ 編譯錯誤:\n{e}"

    try:
        output = subprocess.check_output([os.path.join(TEMP_BIN_DIR, "temp")], cwd=TEMP_BIN_DIR)
        return output.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"執行錯誤:\n{e}"

def compile_and_run(c_code, example_code):
    """
    將 C 程式碼編譯並執行,返回輸出結果
    """
    # 如果選擇了範例程式碼,則使用範例程式碼
    if example_code:
        c_code = example_code

    temp_c_file = os.path.join(TEMP_BIN_DIR, "temp.cpp")
    temp_bin_file = os.path.join(TEMP_BIN_DIR, "temp")

    with open(temp_c_file, "w") as f:
        f.write(c_code)

    try:
        subprocess.run(["g++", "-o", temp_bin_file, temp_c_file], cwd=TEMP_BIN_DIR, check=True)
    except subprocess.CalledProcessError as e:
        return f"g++ 編譯錯誤:\n{e}"

    try:
        output = subprocess.check_output([os.path.join(TEMP_BIN_DIR, "temp")], cwd=TEMP_BIN_DIR)
        return output.decode().strip()
    except subprocess.CalledProcessError as e:
        return f"執行錯誤:\n{e}"


def on_example_change(example_code):
    """
    當選擇範例程式碼時,將程式碼填充到文本框中
    """
    return example_code

with gr.Blocks() as app:
    with gr.Row():
        c_code = gr.Code(label="C 程式碼", language="shell")
    with gr.Row():
        example_code = gr.Dropdown(EXAMPLE_CODES, label="選擇範例程式", interactive=True)
    with gr.Row():
        output = gr.HTML(label="輸出結果")
    with gr.Row():
        run_button = gr.Button("Run")
    example_code.change(fn=on_example_change, inputs=example_code, outputs=c_code)
    run_button.click(compile_and_run, inputs=[c_code, example_code], outputs=[output])

app.launch()