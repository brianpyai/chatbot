from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from typing import List
import sqlite3
import os

# 數據庫設置
DATABASE_NAME = "crm.db"
# Pydantic模型
class CustomerBase(BaseModel):
    name: str
    email: str

class CustomerCreate(CustomerBase):
    pass

class CustomerResponse(CustomerBase):
    id: int

class QuotationBase(BaseModel):
    customer_id: int
    description: str
    amount: float

class QuotationCreate(QuotationBase):
    pass

class QuotationResponse(QuotationBase):
    id: int
    created_at: datetime

class InvoiceBase(BaseModel):
    customer_id: int
    description: str
    amount: float

class InvoiceCreate(InvoiceBase):
    pass

class InvoiceResponse(InvoiceBase):
    id: int
    created_at: datetime
def get_db():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# 檢查並創建必要的表格
def check_and_create_tables():
    conn = get_db()
    cursor = conn.cursor()
    
    # 檢查 customers 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='customers'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )
        ''')
    
    # 檢查 quotations 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quotations'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE quotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    # 檢查 invoices 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='invoices'")
    if not cursor.fetchone():
        cursor.execute('''
            CREATE TABLE invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    
    conn.commit()
    conn.close()

# Updated HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRM System</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1, h2 { color: #333; }
        form { margin-bottom: 20px; }
        input, textarea { margin-bottom: 10px; padding: 5px; width: 300px; }
        input[type="submit"] { width: auto; cursor: pointer; }
        #result, #allContent, #sqlResult { margin-top: 20px; padding: 10px; background-color: #f0f0f0; border: 1px solid #ddd; }
        .error { color: red; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>CRM System</h1>
    
    <h2>Menu</h2>
    <ul>
        <li><a href="#customer-form">Add Customer</a></li>
        <li><a href="#quotation-form">Create Quotation</a></li>
        <li><a href="#invoice-form">Create Invoice</a></li>
        <li><a href="#display-all">Display All Content</a></li>
        <li><a href="#sql-query">Execute SQL Query</a></li>
    </ul>

      <h2>Menu</h2>
    <ul>
        <li><a href="#customer-form">Add Customer</a></li>
        <li><a href="#quotation-form">Create Quotation</a></li>
        <li><a href="#invoice-form">Create Invoice</a></li>
    </ul>

    <h2 id="customer-form">Add Customer</h2>
    <form id="customerForm">
        <input type="text" name="name" placeholder="Customer Name" required><br>
        <input type="email" name="email" placeholder="Customer Email" required><br>
        <input type="submit" value="Add Customer">
    </form>

    <h2 id="quotation-form">Create Quotation</h2>
    <form id="quotationForm">
        <input type="number" name="customer_id" placeholder="Customer ID" required><br>
        <textarea name="description" placeholder="Description" required></textarea><br>
        <input type="number" name="amount" placeholder="Amount" step="0.01" required><br>
        <input type="submit" value="Create Quotation">
    </form>

    <h2 id="invoice-form">Create Invoice</h2>
    <form id="invoiceForm">
        <input type="number" name="customer_id" placeholder="Customer ID" required><br>
        <textarea name="description" placeholder="Description" required></textarea><br>
        <input type="number" name="amount" placeholder="Amount" step="0.01" required><br>
        <input type="submit" value="Create Invoice">
    </form>

    <div id="result"></div>

    <h2 id="display-all">Display All Content</h2>
    <button onclick="displayAllContent()">Show All Content</button>
    <div id="allContent"></div>

    <h2 id="sql-query">Execute SQL Query</h2>
    <form id="sqlForm">
        <textarea name="query" placeholder="Enter SQL query" required></textarea><br>
        <input type="submit" value="Execute Query">
    </form>
    <div id="sqlResult"></div>

    <div id="result"></div>

    <script>
       async function handleSubmit(event, url) {
            event.preventDefault();
            const form = event.target;
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());

            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const contentType = response.headers.get("content-type");
                if (contentType && contentType.indexOf("application/json") !== -1) {
                    const result = await response.json();
                    document.getElementById('result').innerHTML = '<h3>Result:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } else {
                    const text = await response.text();
                    throw new Error('Received non-JSON response: ' + text);
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<h3 class="error">Error:</h3><p>' + error.message + '</p>';
            }
        }

        document.getElementById('customerForm').addEventListener('submit', (e) => handleSubmit(e, '/customers/'));
        document.getElementById('quotationForm').addEventListener('submit', (e) => handleSubmit(e, '/quotations/'));
        document.getElementById('invoiceForm').addEventListener('submit', (e) => handleSubmit(e, '/invoices/'));

        async function displayAllContent() {
            try {
                const response = await fetch('/all_content');
                const data = await response.json();
                let html = '<h3>All Content:</h3>';
                for (const [table, rows] of Object.entries(data)) {
                    html += `<h4>${table}</h4>`;
                    html += '<table><tr>';
                    for (const key in rows[0]) {
                        html += `<th>${key}</th>`;
                    }
                    html += '</tr>';
                    for (const row of rows) {
                        html += '<tr>';
                        for (const value of Object.values(row)) {
                            html += `<td>${value}</td>`;
                        }
                        html += '</tr>';
                    }
                    html += '</table>';
                }
                document.getElementById('allContent').innerHTML = html;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('allContent').innerHTML = '<h3 class="error">Error:</h3><p>' + error.message + '</p>';
            }
        }

        document.getElementById('sqlForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const query = formData.get('query');

            try {
                const response = await fetch('/execute_sql', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({query: query}),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                let html = '<h3>SQL Result:</h3>';
                if (result.length > 0) {
                    html += '<table><tr>';
                    for (const key in result[0]) {
                        html += `<th>${key}</th>`;
                    }
                    html += '</tr>';
                    for (const row of result) {
                        html += '<tr>';
                        for (const value of Object.values(row)) {
                            html += `<td>${value}</td>`;
                        }
                        html += '</tr>';
                    }
                    html += '</table>';
                } else {
                    html += '<p>No results or query executed successfully.</p>';
                }
                document.getElementById('sqlResult').innerHTML = html;
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('sqlResult').innerHTML = '<h3 class="error">Error:</h3><p>' + error.message + '</p>';
            }
        });
    </script>
</body>
</html>
"""

# FastAPI應用
app = FastAPI()

# 在應用啟動時檢查並創建表格
@app.on_event("startup")
async def startup_event():
    check_and_create_tables()
# New route to get all content
@app.get("/all_content")
def get_all_content():
    conn = get_db()
    cursor = conn.cursor()
    
    tables = ['customers', 'quotations', 'invoices']
    all_data = {}

    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = [dict(row) for row in cursor.fetchall()]
        all_data[table] = rows

    conn.close()
    return all_data

# New route to execute SQL queries
@app.post("/execute_sql")
async def execute_sql(query: str = Form(...)):
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query)
        result = [dict(row) for row in cursor.fetchall()]
        return result
    except sqlite3.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

# API路由
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return HTML_TEMPLATE

@app.post("/customers/", response_model=CustomerResponse)
def create_customer(customer: CustomerCreate):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO customers (name, email) VALUES (?, ?)",
                       (customer.name, customer.email))
        conn.commit()
        new_id = cursor.lastrowid
        return {"id": new_id, "name": customer.name, "email": customer.email}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()

@app.get("/customers/", response_model=List[CustomerResponse])
def read_customers(skip: int = 0, limit: int = 100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers LIMIT ? OFFSET ?", (limit, skip))
    customers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return customers

@app.post("/quotations/", response_model=QuotationResponse)
def create_quotation(quotation: QuotationCreate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO quotations (customer_id, description, amount)
        VALUES (?, ?, ?)
    """, (quotation.customer_id, quotation.description, quotation.amount))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.execute("SELECT * FROM quotations WHERE id = ?", (new_id,))
    new_quotation = dict(cursor.fetchone())
    conn.close()
    return new_quotation

@app.get("/quotations/", response_model=List[QuotationResponse])
def read_quotations(skip: int = 0, limit: int = 100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM quotations LIMIT ? OFFSET ?", (limit, skip))
    quotations = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return quotations

@app.post("/invoices/", response_model=InvoiceResponse)
def create_invoice(invoice: InvoiceCreate):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO invoices (customer_id, description, amount)
        VALUES (?, ?, ?)
    """, (invoice.customer_id, invoice.description, invoice.amount))
    conn.commit()
    new_id = cursor.lastrowid
    cursor.execute("SELECT * FROM invoices WHERE id = ?", (new_id,))
    new_invoice = dict(cursor.fetchone())
    conn.close()
    return new_invoice

@app.get("/invoices/", response_model=List[InvoiceResponse])
def read_invoices(skip: int = 0, limit: int = 100):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices LIMIT ? OFFSET ?", (limit, skip))
    invoices = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return invoices

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)