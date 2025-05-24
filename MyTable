class MyTable {
    constructor(containerElement, initialData = []) {
        this.container = typeof containerElement === 'string' ? document.querySelector(containerElement) : containerElement;
        if (!this.container) {
            console.error("MyTable Error: Container element not found.");
            throw new Error("MyTable container not found.");
        }
        this._data = initialData;
        this.tableElement = null;
        this.theadElement = null;
        this.tbodyElement = null;
        this.isSingleObject = false;

        this.postCallback = (responseData, responseOrError) => {
            if (responseOrError instanceof Error) {
                console.warn("MyTable Post Error:", responseOrError);
            } else {
                console.log("MyTable Post Success. Response:", responseData);
            }
        };
        this.clickCallback = (event, cellValue, rowIndex, colIdentifier, rowData, cellElement) => {};
        this.editCallback = (newValue, oldValue, rowIndex, colIdentifier, rowData, cellElement) => {
            let typedNewValue = newValue;
            const originalDataType = typeof oldValue;

            if (originalDataType === 'number') {
                typedNewValue = parseFloat(newValue);
                if (isNaN(typedNewValue)) typedNewValue = oldValue;
            } else if (originalDataType === 'boolean') {
                if (newValue.toLowerCase() === 'true') typedNewValue = true;
                else if (newValue.toLowerCase() === 'false') typedNewValue = false;
                else typedNewValue = oldValue;
            }

            if (this.isSingleObject) {
                if (this._data.hasOwnProperty(colIdentifier)) {
                    this._data[colIdentifier] = typedNewValue;
                }
            } else if (Array.isArray(this._data) && this._data[rowIndex]) {
                if (Array.isArray(this._data[rowIndex])) {
                    this._data[rowIndex][colIdentifier] = typedNewValue;
                } else if (typeof this._data[rowIndex] === 'object' && this._data[rowIndex] !== null) {
                    this._data[rowIndex][colIdentifier] = typedNewValue;
                }
            }
        };
        this.cellRenderCallback = null;
        this.headerRenderCallback = null;
        this.rowRenderCallback = null;

        this._styleId = `my-table-styles-${Math.random().toString(36).substring(2, 11)}`;

        this.init();
        if (initialData) this.render();
    }

    init() {
        this.container.innerHTML = '';
        this.tableElement = document.createElement('table');
        this.tableElement.classList.add('my-table');
        this.theadElement = this.tableElement.createTHead();
        this.tbodyElement = this.tableElement.createTBody();
        this.container.appendChild(this.tableElement);
    }

    get data() {
        return this._data;
    }

    set data(newData) {
        this.isSingleObject = typeof newData === 'object' && newData !== null && !Array.isArray(newData);
        this._data = newData;
        this.render();
    }

    css(cssString) {
        let styleElement = document.getElementById(this._styleId);
        if (!styleElement) {
            styleElement = document.createElement('style');
            styleElement.id = this._styleId;
            document.head.appendChild(styleElement);
        }
        styleElement.textContent = cssString;
        return this;
    }

    renderCellContent(cellElement, value, isEditable = true) {
        cellElement.innerHTML = ''; // Clear previous content
        cellElement.setAttribute('contenteditable', 'false'); // Default to not editable

        // Allow user to override rendering completely
        if (this.cellRenderCallback) {
            this.cellRenderCallback(value, cellElement._myTableContext.rowIndex, cellElement._myTableContext.colIdentifier, cellElement, cellElement._myTableContext.rowData);
            return;
        }

        if (typeof value === 'string') {
            let youtubeVideoId = null;
            // Robust YouTube ID extraction
            if (value.includes("youtube.com/watch")) { // Handles watch?v=... and watch_popup?v=... etc.
                try { youtubeVideoId = new URL(value).searchParams.get('v'); } catch (e) {/* ignore */}
            } else if (value.includes("youtube.com/embed/")) {
                youtubeVideoId = value.split("embed/")[1]?.split(/[?&]/)[0];
            } else if (value.includes("youtu.be/")) {
                youtubeVideoId = value.split("youtu.be/")[1]?.split(/[?&]/)[0];
            }

            if (youtubeVideoId) {
                const iframe = document.createElement('iframe');
                iframe.src = `https://www.youtube.com/embed/${youtubeVideoId}`;
                iframe.width = '160'; iframe.height = '90';
                iframe.frameBorder = '0'; iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"; iframe.allowFullscreen = true;
                cellElement.appendChild(iframe);
            }
            // Image handling (common extensions + base64)
            else if (value.match(/\.(jpeg|jpg|gif|png|svg|webp|bmp|ico)$/i) || value.startsWith('data:image')) {
                const img = document.createElement('img');
                img.src = value;
                img.style.maxWidth = '100px'; img.style.maxHeight = '70px'; img.style.display = 'block';
                img.onerror = () => {
                    cellElement.textContent = 'Error loading image';
                    // if (isEditable) cellElement.setAttribute('contenteditable', 'true'); // Allow editing broken link
                };
                cellElement.appendChild(img);
            }
            // Video handling (common extensions + base64)
            else if (value.match(/\.(mp4|webm|ogg|mov|avi|mkv)$/i) || value.startsWith('data:video')) {
                const video = document.createElement('video');
                video.src = value; video.controls = true; video.style.maxWidth = '150px';  video.style.maxHeight = '100px';
                video.onerror = () => { cellElement.textContent = 'Error loading video'; };
                cellElement.appendChild(video);
            }
            // Audio handling (common extensions + base64)
            else if (value.match(/\.(mp3|wav|aac|oga|flac|m4a)$/i) || value.startsWith('data:audio')) {
                const audio = document.createElement('audio');
                audio.src = value; audio.controls = true; audio.style.width = '100%';
                audio.onerror = () => { cellElement.textContent = 'Error loading audio'; };
                cellElement.appendChild(audio);
            }
            // General web links (if not matched above)
            else if ((value.startsWith('http://') || value.startsWith('https://')) && value.includes('.')) { // Basic check for a TLD
                const a = document.createElement('a');
                a.href = value; a.textContent = value.length > 30 ? value.substring(0, 27) + '...' : value;
                a.target = '_blank';
                cellElement.appendChild(a);
                // If the link text itself should be editable, set contenteditable="true" here.
                // For now, the link itself is not editable, but the original data string would be if it were plain text.
            }
            // Plain text content
            else {
                cellElement.textContent = value;
                if (isEditable) cellElement.setAttribute('contenteditable', 'true');
            }
        } else if (typeof value === 'boolean') {
            cellElement.textContent = value.toString();
            if (isEditable) cellElement.setAttribute('contenteditable', 'true');
        } else if (value instanceof HTMLElement) {
            cellElement.appendChild(value); // Append DOM element directly
        } else if (value === null || value === undefined) {
            cellElement.textContent = ''; // Represent null/undefined as empty string
            if (isEditable) cellElement.setAttribute('contenteditable', 'true');
        } else { // Numbers, other primitives
            cellElement.textContent = String(value);
            if (isEditable) cellElement.setAttribute('contenteditable', 'true');
        }
    }

    render() {
        if (!this.tableElement) this.init();
        this.theadElement.innerHTML = '';
        this.tbodyElement.innerHTML = '';
        this.tableElement.caption && this.tableElement.caption.remove();

        if (this._data === null || this._data === undefined) {
            const cap = this.tableElement.createCaption(); cap.textContent = "Data is null or undefined.";
            return;
        }

        if (this.isSingleObject) {
            this.renderSingleObject();
            return;
        }

        if (!Array.isArray(this._data) || this._data.length === 0) {
            const cap = this.tableElement.createCaption(); cap.textContent = "No data to display or data is not an array.";
            return;
        }

        let headers = [];
        const firstItem = this._data[0];

        if (typeof firstItem === 'object' && firstItem !== null && !Array.isArray(firstItem)) {
            headers = Object.keys(firstItem);
        } else if (Array.isArray(firstItem)) {
            headers = firstItem.map((_, i) => `Column ${i + 1}`); // Placeholder headers for array of arrays
        } else { // Array of primitives
            headers = ['Value'];
        }

        const headerRow = this.theadElement.insertRow();
        headers.forEach((headerText, colIndex) => {
            let th = document.createElement('th');
            if (this.headerRenderCallback) {
                this.headerRenderCallback(headerText, colIndex, th);
            } else {
                th.textContent = headerText;
            }
            headerRow.appendChild(th);
        });

        this._data.forEach((rowDataItem, rowIndex) => {
            let tr = this.tbodyElement.insertRow();
            tr._myTableContext = { rowIndex, rowData: rowDataItem };

            if (this.rowRenderCallback) {
                this.rowRenderCallback(rowDataItem, rowIndex, tr);
            }

            headers.forEach((headerKeyOrIndexBase, colIndex) => {
                const td = tr.insertCell();
                let cellValue;
                let colIdentifier;
                let isEditableCell = true; // Default for the cell, renderCellContent might override for specific content types

                if (typeof rowDataItem === 'object' && rowDataItem !== null && !Array.isArray(rowDataItem)) {
                    colIdentifier = headers[colIndex];
                    cellValue = rowDataItem[colIdentifier];
                } else if (Array.isArray(rowDataItem)) {
                    colIdentifier = colIndex;
                    cellValue = rowDataItem[colIdentifier];
                } else {
                    colIdentifier = 0; // For array of primitives
                    cellValue = rowDataItem;
                }

                td._myTableContext = { rowIndex, colIdentifier, rowData: rowDataItem, originalValue: cellValue };
                this.renderCellContent(td, cellValue, isEditableCell);

                td.addEventListener('click', (event) => {
                    this.clickCallback(event, td._myTableContext.originalValue, rowIndex, colIdentifier, rowDataItem, td);
                });

                td.addEventListener('blur', (event) => {
                    if (event.target.getAttribute('contenteditable') === 'true') {
                        const cellElement = event.target;
                        let newValue = cellElement.textContent; // For simple text.
                        // If cell contains complex HTML (e.g. from cellRenderCallback), this might need adjustment
                        // or the callback itself should handle updates.

                        const oldValue = td._myTableContext.originalValue;

                        if (newValue !== String(oldValue)) {
                           this.editCallback(newValue, oldValue, rowIndex, colIdentifier, rowDataItem, td);
                           // Update originalValue in context after editCallback might have typed it
                           if (this.isSingleObject) {
                               td._myTableContext.originalValue = this._data[colIdentifier];
                           } else if (Array.isArray(this._data[rowIndex])) {
                               td._myTableContext.originalValue = this._data[rowIndex][colIdentifier];
                           } else if (this._data[rowIndex]){
                               td._myTableContext.originalValue = this._data[rowIndex][colIdentifier];
                           }
                        }
                    }
                });
            });
        });
    }

    renderSingleObject() {
        this.theadElement.innerHTML = '';
        this.tbodyElement.innerHTML = '';

        const headerRow = this.theadElement.insertRow();
        ['Property', 'Value'].forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            headerRow.appendChild(th);
        });

        Object.entries(this._data).forEach(([key, value], index) => {
            const tr = this.tbodyElement.insertRow();
            tr.insertCell().textContent = key;
            const tdValue = tr.insertCell();
            tdValue._myTableContext = { rowIndex: index, colIdentifier: key, rowData: this._data, originalValue: value };

            this.renderCellContent(tdValue, value, true); // Values are generally editable

            tdValue.addEventListener('click', (event) => {
                this.clickCallback(event, tdValue._myTableContext.originalValue, index, key, this._data, tdValue);
            });

            tdValue.addEventListener('blur', (event) => {
                 if (event.target.getAttribute('contenteditable') === 'true') {
                    const newValue = event.target.textContent;
                    const oldValue = tdValue._myTableContext.originalValue;
                    if (newValue !== String(oldValue)) {
                        this.editCallback(newValue, oldValue, index, key, this._data, tdValue);
                        tdValue._myTableContext.originalValue = this._data[key];
                    }
                }
            });
        });
    }

    async post(url, method = 'POST', customHeaders = {}) {
        try {
            const response = await fetch(url, {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    ...customHeaders
                },
                body: JSON.stringify(this._data)
            });
            let responseData;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                responseData = await response.json();
            } else {
                responseData = await response.text();
            }

            if (!response.ok) {
                const error = new Error(`HTTP error! Status: ${response.status}, Body: ${responseData}`);
                error.response = response;
                error.responseData = responseData;
                throw error;
            }
            
            if (this.postCallback) {
                this.postCallback(responseData, response);
            }
            return responseData;
        } catch (error) {
            console.error("MyTable.post error:", error);
            if (this.postCallback) {
                this.postCallback(error.responseData, error);
            }
            throw error;
        }
    }
}
