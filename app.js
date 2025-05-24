document.addEventListener('DOMContentLoaded', () => {
    const narrationArea = document.getElementById('narrationArea');
    const tableContainer = document.getElementById('myTableContainer1');
    const searchInput = document.getElementById('searchInput');
    const btnPostData = document.getElementById('btnPostData');
    const chartDisplayArea = document.getElementById('chartDisplayArea');
    const barChartDiv = document.getElementById('barChart');

    let myTable;
    let originalData = [];

    function showNarration(htmlContent, step = "") {
        narrationArea.innerHTML = `<p><strong>Step ${step}:</strong> ${htmlContent}</p>`;
        narrationArea.classList.add('visible');
        narrationArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    function disableAllControls(except = []) {
        document.querySelectorAll('.controls button, .controls input').forEach(el => {
            if (!except.includes(el)) {
                 el.disabled = true;
            }
        });
    }

    function enableControls(elements = []) {
         elements.forEach(el => el.disabled = false);
    }

    function initDemo() {
        tableContainer.innerHTML = '<h2>Demonstration Table</h2><p style="text-align:center; color:#7f8c8d;">Table will appear here.</p>';
        searchInput.value = '';
        searchInput.disabled = true;
        btnPostData.disabled = true;
        chartDisplayArea.style.display = 'none';
        barChartDiv.innerHTML = '';
        if (myTable && myTable.tableElement) {
            const oldStyle = document.getElementById(myTable._styleId);
            if(oldStyle) oldStyle.remove();
        }
        myTable = new MyTable('#myTableContainer1');

        myTable.clickCallback = (event, cellValue, rowIndex, colIdentifier, rowData, cellElement) => {
            console.log("Cell Clicked:", { cellValue, rowIndex, colIdentifier, rowData });
            showNarration(`Cell clicked! Row: ${rowIndex}, Column/Key: ${colIdentifier}, Value: ${cellValue}. This is using <strong>myTable.clickCallback</strong>.`, "Info");
        };
        
        myTable.editCallback = (newValue, oldValue, rowIndex, colIdentifier, rowData, cellElement) => {
            let finalValue = newValue;
            const originalDataType = typeof oldValue;

            if (originalDataType === 'number') {
                finalValue = parseFloat(newValue);
                if (isNaN(finalValue)) finalValue = oldValue;
            } else if (originalDataType === 'boolean') {
                if (newValue.toLowerCase() === 'true') finalValue = true;
                else if (newValue.toLowerCase() === 'false') finalValue = false;
                else finalValue = oldValue;
            }
            
            if (myTable.isSingleObject) {
                 if (myTable.data.hasOwnProperty(colIdentifier)) myTable.data[colIdentifier] = finalValue;
            } else if (Array.isArray(myTable.data) && myTable.data[rowIndex]) {
                if (Array.isArray(myTable.data[rowIndex])) {
                    myTable.data[rowIndex][colIdentifier] = finalValue;
                } else if (typeof myTable.data[rowIndex] === 'object' && myTable.data[rowIndex] !== null) {
                    myTable.data[rowIndex][colIdentifier] = finalValue;
                }
            }
            showNarration(`Data edited! Row: ${rowIndex}, Column/Key: ${colIdentifier}. Old: "${oldValue}", New: "${newValue}" (processed as ${finalValue}). This uses <strong>myTable.editCallback</strong>. The <strong>myTable.data</strong> is updated.`, "Info");
            console.log("Data Edited (app.js):", { newValue, oldValue, finalValue, rowIndex, colIdentifier, rowData });
        };

        showNarration("Welcome! Click a button to start. The table will appear below.", "Start");
        enableControls(Array.from(document.querySelectorAll('.controls button')));
        searchInput.disabled = true;
        btnPostData.disabled = true;
    }

    document.getElementById('btnReset').addEventListener('click', initDemo);

    document.getElementById('btnLoadBasic').addEventListener('click', () => {
        originalData = [
            { id: 1, name: "Alice Wonderland", age: 30, city: "New York" },
            { id: 2, name: "Bob The Builder", age: 24, city: "London" },
            { id: 3, name: "Charlie Chaplin", age: 45, city: "Paris" },
            { id: 4, name: "Diana Prince", age: 28, city: "Berlin" }
        ];
        myTable.data = originalData;
        showNarration("Loaded basic data. MyTable automatically creates headers and rows. Try searching now!", "1. Basic Data");
        searchInput.disabled = false;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
        myTable.cellRenderCallback = null;
    });

    document.getElementById('btnEditable').addEventListener('click', () => {
        originalData = [
            { task: "Buy groceries", status: "Pending", priority: "High", editableNotes: "Milk, Eggs, Bread" },
            { task: "Pay bills", status: "Done", priority: "High", editableNotes: "Electricity bill paid" },
            { task: "Gym session", status: "Pending", priority: "Medium", editableNotes: "Leg day" }
        ];
        myTable.data = originalData;
        showNarration("Cells are editable by default for simple text. Click on a cell in 'editableNotes' or 'status' and try changing its content. Press Enter or click outside to save. The <strong>myTable.editCallback</strong> is triggered.", "2. Editable Cells");
        searchInput.disabled = false;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
        myTable.cellRenderCallback = null;
    });

    document.getElementById('btnLoadMultimedia').addEventListener('click', () => {
        originalData = [
            { item: "PNG Logo", type: "Image URL", source: "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png" },
            { item: "WebP Image", type: "Image URL", source: "https://www.gstatic.com/webp/gallery/1.webp" },
            { item: "Base64 GIF", type: "Image Base64", source: "data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs=" }, // Tiny red dot
            { item: "MP4 Video", type: "Video URL", source: "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" },
            { item: "MP3 Audio", type: "Audio URL", source: "https://interactive-examples.mdn.mozilla.net/media/cc0-audio/t-rex-roar.mp3" },
            { item: "External Link", type: "Webpage", source: "https://www.wikipedia.org" },
            { item: "Broken Image", type: "Image URL", source: "https://example.com/nonexistentimage.png"},
            { item: "Plain Text", type: "Text", source: "This is just some text."}
        ];
        myTable.data = originalData;
        showNarration("Loaded multimedia data. MyTable automatically detects and displays various image types (PNG, WebP, Base64 GIF), videos, and audio files from URLs. Broken media links show an error message. Plain text is displayed as is.", "3. Multimedia");
        searchInput.disabled = false;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
        myTable.cellRenderCallback = null;
    });

    document.getElementById('btnLoadYouTube').addEventListener('click', () => {
        originalData = [
            { title: "The Power of Long-Term Planning: Creating a 5-10 Year Vision", video_url: "https://youtu.be/1HsEG7EjJK4?si=-tNlPM4xtkGOROyL" },
            { title: "Why Do We Need to Explore the Unknown?", video_url: "https://youtu.be/7mtVhrWUGqw?si=ZUcOq3bIY0VSzMKR" },
            { title: "The Interconnectedness of All Things in the Universe", video_url: "https://youtu.be/JQWhyPesh7k?si=FJxCWGoGHQ2hZT8r" },
            { title: "What Are Advanced Thinking Techniques?", video_url: "https://youtu.be/UaflckujFQA?si=DMGuV93ui4of_YTX" },
            { title: "Why Tools Matter in Design", video_url: "https://youtu.be/OOMyLEO_QZU?si=w5YqN4bGBrxErclx" }
        ];
        myTable.data = originalData;
        showNarration("Loaded YouTube video links. MyTable now robustly embeds videos from various YouTube URL formats (watch, embed, youtu.be, with parameters).", "4. YouTube Videos");
        searchInput.disabled = false;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
        myTable.cellRenderCallback = null;
    });

    document.getElementById('btnLoadFormData').addEventListener('click', () => {
        originalData = [
            { field: "Username", value: "JohnDoe", type: "text" },
            { field: "Email", value: "john.doe@example.com", type: "email" },
            { field: "Subscribed", value: true, type: "boolean" },
            { field: "Bio", value: "Loves coding and MyTable!", type:"textarea"}
        ];
        myTable.data = originalData;
        myTable.cellRenderCallback = (cellValue, rowIndex, colIdentifier, cellElement, rowData) => {
            cellElement.innerHTML = '';
            if (colIdentifier === 'value') {
                if (rowData.type === 'boolean') {
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.checked = !!cellValue; // Ensure boolean
                    checkbox.addEventListener('change', (e) => {
                        const newCheckedState = e.target.checked;
                        // Manually call editCallback because we're using a custom input
                        myTable.editCallback(newCheckedState, cellValue, rowIndex, colIdentifier, rowData, cellElement);
                        // The default editCallback will update myTable._data
                    });
                    cellElement.appendChild(checkbox);
                    cellElement.setAttribute('contenteditable', 'false');
                } else if (rowData.type === 'textarea'){
                    const textarea = document.createElement('textarea');
                    textarea.style.width = '95%';
                    textarea.style.minHeight = '40px';
                    textarea.value = cellValue;
                    textarea.addEventListener('blur', (e) => {
                         myTable.editCallback(e.target.value, cellValue, rowIndex, colIdentifier, rowData, cellElement);
                    });
                    cellElement.appendChild(textarea);
                    cellElement.setAttribute('contenteditable', 'false');
                }
                else {
                    cellElement.textContent = cellValue;
                    cellElement.setAttribute('contenteditable', 'true');
                }
            } else {
                cellElement.textContent = cellValue;
                cellElement.setAttribute('contenteditable', 'false');
            }
        };
        showNarration("Loaded form-like data. The 'value' column is editable. The 'Subscribed' field uses a custom <strong>cellRenderCallback</strong> to show a checkbox. Try editing values and then 'Post Data'.", "5. Form Data");
        searchInput.disabled = true;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
    });

    btnPostData.addEventListener('click', async () => {
        if (!myTable || !myTable.data) {
            showNarration("No data to post.", "Error");
            return;
        }
        showNarration("Posting data... (mocking a server request). Check the console for payload. The <strong>myTable.post(url)</strong> method is used. A <strong>postCallback</strong> handles the response.", "6. Post Data");
        
        const mockUrl = "https://jsonplaceholder.typicode.com/posts";

        myTable.postCallback = (responseData, responseOrError) => {
            if (responseOrError instanceof Error) {
                 showNarration(`Post failed! Error: ${responseOrError.message}. Check console for details.`, "Post Response");
                 console.error("Actual Post Error:", responseOrError);
                 console.error("Response Data (if any):", responseOrError.responseData);
            } else {
                showNarration(`Data posted successfully (mocked)! Server responded (see console). Status: ${responseOrError.status}. You could use this response to update the table.`, "Post Response");
                console.log("Mock Post Response Data:", responseData);
                console.log("Full Response Object:", responseOrError);
            }
        };

        try {
            await myTable.post(mockUrl);
        } catch (e) { /* Handled by postCallback */ }
    });

    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        if (!searchTerm) {
            myTable.data = originalData;
            return;
        }
        // Ensure originalData is an array before filtering
        if (!Array.isArray(originalData)) {
            console.warn("Search attempted on non-array data. Resetting to empty.");
            myTable.data = [];
            return;
        }
        const filteredData = originalData.filter(row => {
            // Ensure row is an object before trying Object.values
            if (typeof row !== 'object' || row === null) return false;
            return Object.values(row).some(value =>
                String(value).toLowerCase().includes(searchTerm)
            );
        });
        myTable.data = filteredData;
    });

    document.getElementById('btnApplyCustomCSS').addEventListener('click', () => {
        const customCSS = `
            .my-table {
                border: 3px solid #e67e22;
                font-family: 'Courier New', Courier, monospace;
            }
            .my-table th {
                background-color: #2c3e50; color: #e67e22;
                text-transform: uppercase; letter-spacing: 1px;
            }
            .my-table td { padding: 12px; background-color: #34495e; }
            .my-table tr:nth-child(even) td { background-color: #3b5166; }
            .my-table tr:hover td { background-color: #4a6278; color: #f1c40f; }
            .my-table td[contenteditable="true"]:focus { background-color: #f1c40f; color: #2c3e50; }
        `;
        myTable.css(customCSS);
        showNarration("Applied custom CSS using <strong>myTable.css()</strong>. The table's appearance is now dramatically different!", "7. Custom CSS");
    });

    document.getElementById('btnShowChartData').addEventListener('click', () => {
        originalData = [
            { category: "Alpha", value: 20, color: "#3498db" },
            { category: "Bravo", value: 55, color: "#e74c3c" },
            { category: "Charlie", value: 30, color: "#2ecc71" },
            { category: "Delta", value: 75, color: "#f1c40f" },
            { category: "Echo", value: 40, color: "#9b59b6" }
        ];
        myTable.data = originalData;
        myTable.clickCallback = (event, cellValue, rowIndex, colIdentifier, rowData) => {
            if (colIdentifier === 'value' || colIdentifier === 'category') {
                showNarration(`Clicked on chart data: Category ${rowData.category}, Value ${rowData.value}. Displaying a simple bar chart below.`, "Info");
                displaySimpleBarChart(rowData);
            } else {
                 console.log("Cell Clicked:", { cellValue, rowIndex, colIdentifier, rowData });
                 showNarration(`Cell clicked! Row: ${rowIndex}, Column/Key: ${colIdentifier}, Value: ${cellValue}.`, "Info");
            }
        };
        showNarration("Loaded data suitable for charting. Click on a 'value' or 'category' cell in any row to see a very basic bar chart representation of that row's data below the table. This uses <strong>myTable.clickCallback</strong> to trigger external actions.", "8. Chart Data");
        searchInput.disabled = false;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'block';
        barChartDiv.innerHTML = '<p style="text-align:center; color:#7f8c8d;">Click a row in the table above to see a chart.</p>';
        myTable.cellRenderCallback = null;
    });

    function displaySimpleBarChart(rowData) {
        barChartDiv.innerHTML = '';
        chartDisplayArea.style.display = 'block';
        const maxChartValue = 100;

        const wrapper = document.createElement('div');
        wrapper.className = 'bar-wrapper';
        const bar = document.createElement('div');
        bar.className = 'bar';
        const percentageHeight = (rowData.value / maxChartValue) * 100;
        bar.style.height = Math.max(5, percentageHeight) + '%';
        bar.style.backgroundColor = rowData.color || '#e67e22';
        const barValue = document.createElement('span');
        barValue.className = 'bar-value';
        barValue.textContent = rowData.value;
        const barLabel = document.createElement('div');
        barLabel.className = 'bar-label';
        barLabel.textContent = rowData.category;
        
        wrapper.appendChild(barValue);
        wrapper.appendChild(bar);
        wrapper.appendChild(barLabel);
        barChartDiv.appendChild(wrapper);
    }

    document.getElementById('btnLoadSingleObject').addEventListener('click', () => {
        originalData = {
            appName: "MyTable Suite", version: "1.0.5", releaseDate: "2025-05-23",
            author: "AI Assistant", isStable: true, contact: "support@mytable.com",
            features: "Data Display, Editing, Posting, Multimedia",
            documentationLink: "https://example.com/docs/mytable" // Test a link in single object
        };
        myTable.data = originalData;
        showNarration("Loaded a single JavaScript object. MyTable displays it as a key-value pair table. Values are editable. Multimedia/links in values are also rendered.", "9. Single Object");
        searchInput.disabled = true;
        btnPostData.disabled = false;
        chartDisplayArea.style.display = 'none';
        myTable.cellRenderCallback = null;
    });

    initDemo();
});
