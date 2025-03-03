# 函數式編程在 Python 中的應用：全自動化與 AI 功能的基礎

**追問:**

- 函數式編程如何提升 AI 工作流程的可擴展性？  
- 在 Python 中實現函數式原則的限制是什麼？  
- 模組化設計如何改善複雜系統的自動化能力？

## 正文

函數式編程強調不可變性、純粹性和函數組合，為系統設計提供堅實的基礎。雖然 Python 非純函數式語言，但其靈活性支持這些原則，從而實現模組化、可擴展且高效的解決方案。本文探討 AI 與自動化中的三大組件：全自動短片生成工具、圖片存儲系統（已更新新程式碼）和日誌記錄系統，用實際程式碼展示其功能和優勢。

### 函數式編程的關鍵特性

- **純粹性：** 函數完全依賴輸入，杜絕副作用。  
- **不可變性：** 資料保持不變，修改時生成全新結構。  
- **函數組合：** 將小函數組合作為解決複雜問題的工具。  
- **高階函數：** 調用、傳遞或作為返回值使用其他函數。  
- **模組化與抽象化：** 將代碼拆分成可重用的獨立單元。

這些特性支撐可靠且高效的系統架構。

---

## 實例分析

### 1. 全自動短片生成工具

該工具利用 AI 生成帶特效和音樂的動畫短片，展示了函數式設計的力量。以下是一個核心函數 `makeCharater`，該函數通過多個模組化步驟完成影片生成任務：

```python
def makeCharater(character="Lovely", targetDir=Gbase, targetName=None, n=35, scale=0.25, 
                 animationDuration=3000, tagConditions=TagConditions, target_width=648, 
                 target_height=1152, M=True):
    """
    透過下列步驟生成角色影片：
      1. 從圖片庫中提取 gallery 與 prompts。
      2. 設定影片名稱與圖片輸出路徑。
      3. 利用 ConceptNetwork 重排 gallery。
      4. 將圖片儲存至指定資料夾，並嘗試去背處理。
      5. 生成動畫，加入動態文字與特效。
      6. 影片生成後添加背景音樂。
    """
    gallery, prompts = get_gallery_and_prompts(character, n)
    targetPath, targetImagesPath, targetName0 = prepare_target_names(character, targetDir, targetName, n)
    if os.path.exists(targetPath):
        print(f"{targetPath} 已存在，退出。")
        return
    gallery = reorder_gallery(gallery, prompts, character)
    gallery = save_gallery_images(gallery, targetImagesPath)
    animator = MyVideosWriter(targetPath, gallery, scale=scale, width=target_width, height=target_height)
    concept_text = gentTextDir(targetImagesPath)
    all_images, adjusted_nobg_images = get_adjusted_images(targetImagesPath, target_width, target_height)
    generate_video_frames(animator, character, targetName0, targetImagesPath, 
                          target_width, target_height, concept_text, adjusted_nobg_images, all_images)
    animator.process_video_frames()
    print(f"影片處理完成，儲存至 {targetPath}")
    if M:
        output_video_path = MyMusicGenerator.addMusicToVideo(targetPath, tagConditions=tagConditions)
        print(f"音樂已加入 {output_video_path}")
```

**功能亮點：**

- **模組化：** 每一步驟由獨立函數處理（如 `get_gallery_and_prompts`、`save_gallery_images`）；降低耦合，提高自動化處理效率。  
- **不可變性：** 像 `gallery` 的數據在多個函數間傳遞但不被改變，確保數據一致性。  
- **抽象化：** 像 `MyVideosWriter` 這類封裝工具將複雜性隱藏，讓主要邏輯保持簡潔。

---

### 2. 圖片存儲系統

本系統基於 SQLite3 架構的 `FileDict` 和 `FileSQL3`，設計用以高效、可靠且節省內存地存儲數十萬張圖片及相關資料。以下是更新後的核心程式碼：

```python
from pathlib import Path
import os
import json

from tempCharatersP import tempCharatersDescription, tempCharaters
from fileDict3 import FileDict, FileSQL3

import platform
delTasks = []

def list_images(directory):
    """
    遍歷指定目錄，返回所有圖片文件（擴展名為 jpg, jpeg, png, bmp, svg, webp）的路徑列表。
    """
    valid_extensions = {"jpg", "jpeg", "png", "bmp", "svg", "webp"}
    images = []
    try:
        for entry in os.scandir(directory):
            if entry.is_file() and entry.name.split(".")[-1].lower() in valid_extensions:
                images.append(entry.path)
    except Exception as e:
        # 根據需要記錄日誌，這裡默認忽略異常
        print(f"Error scanning directory {directory}: {e}")
    return images

def delete_all_images(directory):
    ls = list_images(directory)
    total = len(ls)
    if not total:
        return
    for i, path in enumerate(ls):
        print(f"Deleting {i+1}/{total}: {path}")
        os.remove(path)

def process_character(character, images_db, keys_db, temp_db):
    global delTasks

    """
    處理單個角色相關目錄：
      - 掃描角色目錄中所有圖片文件
      - 將圖片存入 images_db，並更新 keys_db（新增圖片檔名）
      - 完成後刪除該目錄中的所有圖片
      - 同時處理角色名稱後綴 "0" 的目錄存入 temp_db
    返回處理的總圖片數量。
    """
    # 嘗試讀取現有的圖片 keys 列表
    if character in keys_db:
        try:
            character_keys = json.loads(keys_db[character])
        except Exception:
            character_keys = []
    else:
        character_keys = []

    processed_count = 0
    image_paths = list_images(character)
    print(f"Processing '{character}'，找到 {len(image_paths)} 張圖片。")

    for i, image_full_path in enumerate(image_paths):
        p_path = Path(image_full_path).name
        if p_path in character_keys:
            continue
        try:
            images_db.put(file_path=image_full_path, p_path=p_path, commit=False)
            print(f"{i}: {image_full_path} 處理完成並加入資料庫。")
            character_keys.append(p_path)
            processed_count += 1
        except Exception as e:
            print(f"處理 {image_full_path} 時發生錯誤: {e}")

    try:
        keys_db[character] = json.dumps(character_keys)
        keys_db._commit()
        print(f"角色 '{character}' 的 keys 更新成功。")
    except Exception as e:
        print(f"更新 {character} 的 keys_db 時發生錯誤: {e}")

    images_db.conn.commit()
    if len(image_paths) > 0:
        delTasks.append(character)
    
    # 處理角色名稱後綴 "0" 的目錄
    secondary_directory = character + "0"
    secondary_paths = list_images(secondary_directory)
    print(f"Processing '{secondary_directory}'，找到 {len(secondary_paths)} 張圖片。")
    processed_secondary = 0
    for i, image_full_path in enumerate(secondary_paths):
        p_path = character + "/" + Path(image_full_path).name
        try:
            temp_db.put(file_path=image_full_path, p_path=p_path, commit=False)
            print(f"{i}: {image_full_path} 處理完成並加入資料庫。")
            processed_secondary += 1
        except Exception as e:
            print(f"處理 {image_full_path} 時發生錯誤: {e}")

    images_db.conn.commit()
    temp_db.conn.commit()
    
    if processed_secondary > 0:
        delTasks.append(secondary_directory)
    
    return processed_count + processed_secondary

def initialize_databases():
    """
    初始化數據庫對象，返回包含 Allwoman、AllPrompts、AllWomanImagesKeys（使用 FileDict）
    及 AllwomanImages、AllwomanImagesTemp（使用 FileSQL3）的字典。
    """
    databases = {}
    databases["Allwoman"] = FileDict("Allwoman.sql")
    databases["AllPrompts"] = FileDict("Allwoman.sql", table="AllPrompts")
    databases["AllWomanImagesKeys"] = FileDict("Allwoman.sql", table="AllWomanImagesKeys")
    databases["AllwomanImages"] = FileSQL3("AllwomanImages.sql")
    databases["AllwomanImagesTemp"] = FileSQL3("AllwomanImagesTemp.sql")
    return databases

def close_databases(databases):
    """
    關閉所有數據庫連接。
    """
    databases["Allwoman"].close()
    databases["AllPrompts"].close()
    databases["AllWomanImagesKeys"].close()
    databases["AllwomanImages"].close()
    databases["AllwomanImagesTemp"].close()

def main():
    # 初始化數據庫對象
    dbs = initialize_databases()
    
    # 生成角色名稱列表（來自 tempCharaters）
    all_character_names = list(tempCharaters.keys())
    print("AllwomanNames:", all_character_names)

    total_processed = 0

    # 處理每個角色目錄中的圖片
    for character in all_character_names:
        processed = process_character(
            character, 
            images_db=dbs["AllwomanImages"],
            keys_db=dbs["AllWomanImagesKeys"],
            temp_db=dbs["AllwomanImagesTemp"]
        )
        total_processed += processed

    print(f"處理圖片總數: {total_processed}")

    # 關閉所有數據庫連接
    close_databases(dbs)
    for c in delTasks:
        print(f"正在刪除 '{c}' 中的所有圖片 ...")
        delete_all_images(c)
```

**功能亮點：**

- **高效率與穩定性：**  
  基於 SQLite3 的 `FileDict` 和 `FileSQL3` 可高效處理數十萬圖片，且不大量佔用內存。  
- **自動清理：**  
  處理完成後自動刪除原始圖片文件，保持系統存儲整潔。  
- **持續改進：**  
  使用日誌和異常處理監控處理流程，便於後續性能調優和錯誤跟蹤。

---

### 3. 日誌記錄系統

該系統用於記錄實驗數據，支持並發環境，並利用生成器實現懶讀取，確保資源高效使用：

```python
import json, os

def writeLog(data, logsPath):
    """將資料以 JSON 格式追加至日誌文件。"""
    separators = (',', ':')
    data = json.dumps(data, separators=separators)
    with open(logsPath, "a") as Logs:
        Logs.write(data + "\n")

def dataOfLogsFile(logsPath):
    """以懶讀方式逐行讀取文件中的 JSON 記錄。"""
    with open(logsPath) as dataFile:
        while line := dataFile.readline():
            if line:
                yield json.loads(line)

def dataOfLogsDir(logsDir):
    """從指定目錄中所有 .log 檔案依次生成記錄。"""
    for logsFile in os.scandir(logsDir):
        if logsFile.is_file() and logsFile.name.lower().endswith(".log"):
            for data in dataOfLogsFile(logsFile.path):
                yield data
```

**功能亮點：**

- **純粹與簡單：**  
  `writeLog` 專注於資料序列化和寫入，無其他副作用。  
- **懶惰求值：**  
  利用生成器（`dataOfLogsFile`、`dataOfLogsDir`），可在大數據環境下高效取用記錄。
- **不可變性：**  
  日誌追加保存全部歷史記錄，不會覆蓋已存數據。

---

## 對全自動化與 AI 的基礎支持

這些組件與 AI 工作流程（預處理、推斷與後處理）息息相關，體現了模組化數據驅動設計的諸多優勢：

- **AI 可擴展性：**  
  模組化設計允許具體功能如 `list_images`、圖片儲存處理等可並行運行，從而高效處理大數據。  
- **Python 限制：**  
  Python 中的可變預設值和全局狀態要求開發者格外留意，確保函數純粹性。  
- **模組化優勢：**  
  將複雜任務拆分成獨立函數（如 `makeCharater` 多步驟處理）使得調試、升級、集成變得更容易。

此外，這些設計帶來靈活性與預測性行為，正是 AI 系統所需的品質。

---

## 結論

通過充分應用函數式編程原則與實用代碼，這些組件展現了模組化、可擴展性與可靠性——這些正是現代全自動化和 AI 系統發展的基石。更新過的圖片存儲系統展示了如何安全高效地處理數十萬圖片，同時不斷在實際應用中進行改進與優化。


#函數式編程 #自動化 #AI生成

