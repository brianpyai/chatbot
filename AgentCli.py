API_KEY= "you key" 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced AskBot with Beautiful ANSI Colors - 美化版增強 AI 對話工具
基於可運作版本，添加選擇、列舉、建議、翻譯、總結、規劃等功能
所有功能返回可解析的 JSON 格式 + 美化的終端輸出
"""


import asyncio
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

try:
    import fastapi_poe as fp
except ImportError:
    print("Error: fastapi_poe not installed. Run: pip install fastapi_poe")
    exit(1)

# ========================================
# 美化輸出系統 - ANSI 色彩
# ========================================

class COLOURS:
    # 基本色彩 (0-15)
    Black = 0
    Maroon = 1
    Green = 2
    Olive = 3
    Navy = 4
    Purple = 5
    Teal = 6
    Silver = 7
    Grey = 8
    Red = 9
    Lime = 10
    Yellow = 11
    Blue = 12
    Fuchsia = 13
    Aqua = 14
    White = 15
    
    # 擴展色彩
    DarkGreen = 22
    DarkBlue = 18
    LightGreen = 120
    LightBlue = 117
    LightYellow = 227
    LightRed = 196
    Orange = 214
    Pink = 218
    Cyan = 51
    Magenta = 201

class STYLES:
    Bold = "Bold"
    Dim = "Dim"
    Italic = "Italic"
    Underline = "Underline"
    SlowBlink = "SlowBlink"
    FastBlink = "FastBlink"
    Reverse = "Reverse"
    Hidden = "Hidden"
    Strikethrough = "Strikethrough"

# 建立全局映射字典
StylesDict = {}
for x in dir(COLOURS):
    if not x.startswith("_"):
        StylesDict[x.lower().capitalize()] = x
for x in dir(STYLES):
    if not x.startswith("_"):
        StylesDict[x.lower().capitalize()] = x

class Text:
    ESC = "\033["
    RESET = "\033[0m"
    
    STYLES = {
        "Bold": "1",
        "Dim": "2",
        "Italic": "3",
        "Underline": "4",
        "Slowblink": "5",
        "Fastblink": "6",
        "Reverse": "7",
        "Hidden": "8",
        "Strikethrough": "9"
    }
    
    @classmethod
    def _validate_text(cls, text):
        if not isinstance(text, str):
            raise ValueError("輸入必須是字符串")
    
    @staticmethod
    def _normalize(name: str) -> str:
        if not isinstance(name, str):
            raise ValueError("顏色或樣式名稱必須是字符串")
        return name.lower().capitalize()
    
    @staticmethod
    def color(text, color):
        Text._validate_text(text)
        if isinstance(color, str):
            norm_color = Text._normalize(color)
            if norm_color in StylesDict:
                color_code = getattr(COLOURS, StylesDict[norm_color])
            else:
                return text  # 如果顏色無效，返回原文
        elif isinstance(color, int):
            if 0 <= color <= 255:
                color_code = color
            else:
                return text
        elif isinstance(color, tuple) and len(color) == 3:
            r, g, b = color
            if not all(0 <= c <= 255 for c in (r, g, b)):
                return text
            return f"{Text.ESC}38;2;{r};{g};{b}m{text}{Text.RESET}"
        else:
            return text
        return f"{Text.ESC}38;5;{color_code}m{text}{Text.RESET}"
    
    @staticmethod
    def background(text, bg):
        Text._validate_text(text)
        if isinstance(bg, str):
            norm_bg = Text._normalize(bg)
            if norm_bg in StylesDict:
                bg_code = getattr(COLOURS, StylesDict[norm_bg])
            else:
                return text
        elif isinstance(bg, int):
            if 0 <= bg <= 255:
                bg_code = bg
            else:
                return text
        elif isinstance(bg, tuple) and len(bg) == 3:
            r, g, b = bg
            if not all(0 <= c <= 255 for c in (r, g, b)):
                return text
            return f"{Text.ESC}48;2;{r};{g};{b}m{text}{Text.RESET}"
        else:
            return text
        return f"{Text.ESC}48;5;{bg_code}m{text}{Text.RESET}"
    
    @staticmethod
    def style(text, style_name):
        Text._validate_text(text)
        norm_style = Text._normalize(style_name)
        code = Text.STYLES.get(norm_style)
        if code is None:
            return text
        return f"{Text.ESC}{code}m{text}{Text.RESET}"
    
    @staticmethod
    def apply(text, color=None, background=None, styles=None):
        try:
            Text._validate_text(text)
            codes = []
            
            # 處理樣式
            if styles is not None:
                if not isinstance(styles, (list, tuple)):
                    styles = [styles]
                for st in styles:
                    norm_st = Text._normalize(st)
                    code = Text.STYLES.get(norm_st)
                    if code is not None:
                        codes.append(code)
            
            # 處理前景色
            if color is not None:
                if isinstance(color, str):
                    norm_color = Text._normalize(color)
                    if hasattr(COLOURS, norm_color):
                        color_code = getattr(COLOURS, norm_color)
                        codes.append(f"38;5;{color_code}")
                elif isinstance(color, int):
                    if 0 <= color <= 255:
                        codes.append(f"38;5;{color}")
                elif isinstance(color, tuple) and len(color) == 3:
                    r, g, b = color
                    if all(0 <= c <= 255 for c in (r, g, b)):
                        codes.append(f"38;2;{r};{g};{b}")
            
            # 處理背景色
            if background is not None:
                if isinstance(background, str):
                    norm_bg = Text._normalize(background)
                    if hasattr(COLOURS, norm_bg):
                        bg_code = getattr(COLOURS, norm_bg)
                        codes.append(f"48;5;{bg_code}")
                elif isinstance(background, int):
                    if 0 <= background <= 255:
                        codes.append(f"48;5;{background}")
                elif isinstance(background, tuple) and len(background) == 3:
                    r, g, b = background
                    if all(0 <= c <= 255 for c in (r, g, b)):
                        codes.append(f"48;2;{r};{g};{b}")
            
            if not codes:
                return text
            return f"{Text.ESC}{';'.join(codes)}m{text}{Text.RESET}"
        except:
            return text

# ========================================
# 美化 UI 輔助類
# ========================================

class UI:
    """美化 UI 輔助類"""
    
    @staticmethod
    def success(text):
        """成功訊息 - 綠色 + 粗體"""
        return Text.apply(f"✅ {text}", color=COLOURS.LightGreen, styles=STYLES.Bold)
    
    @staticmethod
    def error(text):
        """錯誤訊息 - 紅色 + 粗體"""
        return Text.apply(f"❌ {text}", color=COLOURS.LightRed, styles=STYLES.Bold)
    
    @staticmethod
    def warning(text):
        """警告訊息 - 黃色 + 粗體"""
        return Text.apply(f"⚠️  {text}", color=COLOURS.Yellow, styles=STYLES.Bold)
    
    @staticmethod
    def info(text):
        """資訊訊息 - 藍色"""
        return Text.apply(f"ℹ️  {text}", color=COLOURS.LightBlue)
    
    @staticmethod
    def processing(text):
        """處理中訊息 - 紫色 + 粗體"""
        return Text.apply(f"🔄 {text}", color=COLOURS.Purple, styles=STYLES.Bold)
    
    @staticmethod
    def question(text):
        """問題訊息 - 青色 + 粗體"""
        return Text.apply(f"❓ {text}", color=COLOURS.Aqua, styles=STYLES.Bold)
    
    @staticmethod
    def answer(text):
        """回答訊息 - 白色 + 粗體"""
        return Text.apply(f"🤖 {text}", color=COLOURS.White, styles=STYLES.Bold)
    
    @staticmethod
    def header(text):
        """標題訊息 - 紫色背景白字"""
        return Text.apply(f" {text} ", color=COLOURS.White, background=COLOURS.Purple, styles=STYLES.Bold)
    
    @staticmethod
    def section(text):
        """段落標題 - 橙色 + 粗體"""
        return Text.apply(f"📋 {text}", color=COLOURS.Orange, styles=STYLES.Bold)
    
    @staticmethod
    def separator(length=60, char="="):
        """分隔線 - 灰色"""
        return Text.apply(char * length, color=COLOURS.Grey, styles=STYLES.Dim)
    
    @staticmethod
    def highlight(text):
        """高亮文字 - 黃色背景黑字"""
        return Text.apply(text, color=COLOURS.Black, background=COLOURS.Yellow)
    
    @staticmethod
    def code(text):
        """代碼文字 - 綠色"""
        return Text.apply(text, color=COLOURS.Green)
    
    @staticmethod
    def json_key(text):
        """JSON 鍵 - 藍色"""
        return Text.apply(f'"{text}"', color=COLOURS.Blue)
    
    @staticmethod
    def json_string(text):
        """JSON 字串值 - 綠色"""
        return Text.apply(f'"{text}"', color=COLOURS.Green)
    
    @staticmethod
    def json_number(text):
        """JSON 數字值 - 黃色"""
        return Text.apply(str(text), color=COLOURS.Yellow)
    
    @staticmethod
    def progress_bar(current, total, width=40):
        """進度條"""
        if total == 0:
            return "📊 " + Text.apply("░" * width, color=COLOURS.Grey) + " 0/0 (0%)"
        
        progress = int(width * current // total)
        percentage = current * 100 // total
        
        filled = Text.apply("█" * progress, color=COLOURS.LightGreen)
        empty = Text.apply("░" * (width - progress), color=COLOURS.Grey)
        
        return f"📊 {filled}{empty} {current}/{total} ({percentage}%)"

# ========================================
# Enhanced AskBot 類
# ========================================

class AskBot:
    """美化版增強 AI 機器人 - 支援多種功能，統一 JSON 返回"""
    
    def __init__(self, api_key: str):
        """初始化機器人"""
        if not api_key or not api_key.strip():
            raise ValueError("API Key 不能為空")
        
        self.api_key = api_key.strip()
        print(UI.success("AskBot 初始化完成"))
    
    async def ask(self, question: str, model: str = "GPT-4.1-mini") -> str:
        """基礎問答功能"""
        if not question.strip():
            return "問題不能為空"
        
        try:
            messages = [fp.ProtocolMessage(role="user", content=question.strip())]
            
            full_response = ""
            async for partial in fp.get_bot_response(
                messages=messages, 
                bot_name=model, 
                api_key=self.api_key
            ):
                full_response += partial.text
            
            return full_response
            
        except Exception as e:
            return f"錯誤: {str(e)}"
    
    def ask_sync(self, question: str, model: str = "GPT-4.1") -> str:
        """同步版本的提問"""
        return asyncio.run(self.ask(question, model))
    
    async def _ask_json(self, prompt: str, model: str = "GPT-4.1-mini") -> Dict:
        """內部方法：要求 AI 返回 JSON 格式"""
        try:
            response = await self.ask(prompt, model)
            # 嘗試解析 JSON
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # 如果無法解析，嘗試提取 JSON 部分
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except:
                        pass
                
                return {
                    "error": "JSON 解析失敗",
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def _pretty_print_json(self, data: Dict, title: str = "結果"):
        """美化打印 JSON 結果"""
        print(f"\n{UI.separator()}")
        print(UI.header(title))
        print(UI.separator())
        
        if "error" in data:
            print(UI.error(f"錯誤: {data['error']}"))
            if "raw_response" in data:
                print(UI.warning("原始回應:"))
                print(data["raw_response"])
        else:
            self._print_json_recursive(data, 0)
    
    def _print_json_recursive(self, obj, indent=0):
        """遞歸打印 JSON 對象"""
        spaces = "  " * indent
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"{spaces}{UI.json_key(key)}:")
                    self._print_json_recursive(value, indent + 1)
                elif isinstance(value, str):
                    print(f"{spaces}{UI.json_key(key)}: {UI.json_string(value)}")
                elif isinstance(value, (int, float)):
                    print(f"{spaces}{UI.json_key(key)}: {UI.json_number(value)}")
                else:
                    print(f"{spaces}{UI.json_key(key)}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                print(f"{spaces}{UI.highlight(f'[{i}]')}:")
                self._print_json_recursive(item, indent + 1)
        else:
            print(f"{spaces}{obj}")
    
    # === 1. 選擇功能 ===
    async def choice(self, options: List[str], criteria: str, model: str = "GPT-4.1-mini") -> Dict:
        """AI 智能選擇功能"""
        print(UI.processing(f"AI 正在分析 {len(options)} 個選項..."))
        
        prompt = f"""請從以下選項中根據給定標準進行選擇：

選項列表：
{json.dumps(options, ensure_ascii=False, indent=2)}

選擇標準：{criteria}

請以純 JSON 格式回答，結構如下：
{{
    "selected_option": "最佳選項",
    "selected_index": 選項索引(0開始),
    "confidence": 信心度(1-10),
    "reasoning": "選擇理由",
    "alternatives": ["其他可行選項"],
    "pros": ["選擇的優點"],
    "cons": ["選擇的缺點"],
    "recommendation": "具體建議"
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "智能選擇結果")
        return result
    
    def choice_sync(self, options: List[str], criteria: str, model: str = "GPT-4.1") -> Dict:
        """同步版本的選擇功能"""
        return asyncio.run(self.choice(options, criteria, model))
    
    # === 2. 列舉功能 ===
    async def enumerate(self, topic: str, count: int = 5, model: str = "GPT-4.1-mini") -> Dict:
        """AI 智能列舉功能"""
        print(UI.processing(f"AI 正在列舉「{topic}」的 {count} 個要點..."))
        
        prompt = f"""請列舉關於「{topic}」的 {count} 個要點。

請以純 JSON 格式回答，結構如下：
{{
    "topic": "主題名稱",
    "total_count": 總數量,
    "items": [
        {{
            "index": 1,
            "title": "要點標題",
            "description": "詳細描述",
            "importance": "重要程度(1-10)",
            "category": "分類",
            "examples": ["相關例子"]
        }}
    ],
    "summary": "整體總結",
    "recommendations": ["行動建議"]
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "智能列舉結果")
        return result
    
    def enumerate_sync(self, topic: str, count: int = 5, model: str = "GPT-4.1") -> Dict:
        """同步版本的列舉功能"""
        return asyncio.run(self.enumerate(topic, count, model))
    
    # === 3. 建議功能 ===
    async def suggest(self, situation: str, goal: str = "", model: str = "GPT-4.1-mini") -> Dict:
        """AI 專業建議功能"""
        print(UI.processing("AI 正在分析情況並生成專業建議..."))
        
        prompt = f"""針對以下情況提供專業建議：

情況描述：{situation}
目標（如有）：{goal}

請以純 JSON 格式回答，結構如下：
{{
    "situation_analysis": "情況分析",
    "key_issues": ["核心問題"],
    "opportunities": ["機會點"],
    "suggestions": [
        {{
            "category": "建議類別",
            "action": "具體行動",
            "priority": "優先級(1-10)",
            "timeline": "建議時程",
            "resources_needed": ["所需資源"],
            "expected_outcome": "預期結果"
        }}
    ],
    "risks": ["潛在風險"],
    "success_metrics": ["成功指標"],
    "next_steps": ["下一步行動"]
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "專業建議結果")
        return result
    
    def suggest_sync(self, situation: str, goal: str = "", model: str = "GPT-4.1") -> Dict:
        """同步版本的建議功能"""
        return asyncio.run(self.suggest(situation, goal, model))
    
    # === 4. 翻譯功能 ===
    async def translate(self, text: str, target_lang: str = "中文", source_lang: str = "auto", model: str = "GPT-4.1-mini") -> Dict:
        """AI 翻譯功能"""
        print(UI.processing(f"AI 正在翻譯 ({source_lang} → {target_lang})..."))
        
        prompt = f"""請將以下文本翻譯：

原文：{text}
源語言：{source_lang}
目標語言：{target_lang}

請以純 JSON 格式回答，結構如下：
{{
    "original_text": "原文",
    "translated_text": "翻譯結果", 
    "source_language": "檢測到的源語言",
    "target_language": "目標語言",
    "confidence": "翻譯信心度(1-10)",
    "alternatives": ["其他翻譯選項"],
    "context_notes": ["語境說明"],
    "word_count": "字數統計"
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "翻譯結果")
        return result
    
    def translate_sync(self, text: str, target_lang: str = "中文", source_lang: str = "auto", model: str = "GPT-4.1") -> Dict:
        """同步版本的翻譯功能"""
        return asyncio.run(self.translate(text, target_lang, source_lang, model))
    
    # === 5. 總結功能 ===
    async def summarize(self, text: str, style: str = "簡潔", model: str = "GPT-4.1-mini") -> Dict:
        """AI 總結功能"""
        print(UI.processing(f"AI 正在總結文本 (風格: {style})..."))
        
        prompt = f"""請總結以下內容：

原文：{text}
總結風格：{style}

請以純 JSON 格式回答，結構如下：
{{
    "original_length": "原文字數",
    "summary_length": "總結字數",
    "compression_ratio": "壓縮比例",
    "main_summary": "主要總結",
    "key_points": ["關鍵要點"],
    "detailed_summary": "詳細總結",
    "categories": ["內容分類"],
    "sentiment": "情感傾向",
    "action_items": ["行動項目"],
    "conclusions": ["結論"]
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "總結結果")
        return result
    
    def summarize_sync(self, text: str, style: str = "簡潔", model: str = "GPT-4.1") -> Dict:
        """同步版本的總結功能"""
        return asyncio.run(self.summarize(text, style, model))
    
    # === 6. 規劃功能 ===
    async def plan(self, goal: str, timeframe: str = "1個月", resources: str = "", model: str = "GPT-4.1-mini") -> Dict:
        """AI 規劃功能"""
        print(UI.processing(f"AI 正在制定規劃 (時程: {timeframe})..."))
        
        prompt = f"""請為以下目標制定執行計劃：

目標：{goal}
時間框架：{timeframe}
可用資源：{resources}

請以純 JSON 格式回答，結構如下：
{{
    "goal": "目標描述",
    "timeframe": "時間框架",
    "phases": [
        {{
            "phase_name": "階段名稱",
            "duration": "持續時間",
            "objectives": ["階段目標"],
            "tasks": [
                {{
                    "task": "具體任務",
                    "priority": "優先級(1-10)",
                    "estimated_time": "預估時間",
                    "dependencies": ["依賴項"],
                    "resources": ["所需資源"]
                }}
            ],
            "deliverables": ["交付成果"],
            "milestones": ["里程碑"]
        }}
    ],
    "risks": ["風險評估"],
    "contingency_plans": ["應急計劃"],
    "success_criteria": ["成功標準"],
    "budget_estimate": "預算估算",
    "recommended_tools": ["推薦工具"]
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        self._pretty_print_json(result, "規劃結果")
        return result
    
    def plan_sync(self, goal: str, timeframe: str = "1個月", resources: str = "", model: str = "GPT-4.1") -> Dict:
        """同步版本的規劃功能"""
        return asyncio.run(self.plan(goal, timeframe, resources, model))
    
    # === 7. 批量處理功能 ===
    async def batch_process(self, items: List[str], operation: str, model: str = "GPT-4.1-mini") -> Dict:
        """批量處理功能"""
        print(UI.processing(f"AI 正在批量處理 {len(items)} 個項目 (操作: {operation})..."))
        
        # 顯示進度
        for i in range(len(items)):
            print(f"\r{UI.progress_bar(i, len(items))}", end="")
            time.sleep(0.1)  # 模擬處理時間
        
        prompt = f"""請對以下項目進行批量{operation}：

項目列表：
{json.dumps(items, ensure_ascii=False, indent=2)}

操作類型：{operation}

請以純 JSON 格式回答，結構如下：
{{
    "operation": "操作類型",
    "total_items": 總項目數,
    "processed_items": [
        {{
            "index": 索引,
            "original": "原始項目",
            "result": "處理結果",
            "status": "成功/失敗",
            "notes": "備註"
        }}
    ],
    "summary": "批量處理總結",
    "success_rate": "成功率"
}}

只返回 JSON，不要其他文字。"""
        
        result = await self._ask_json(prompt, model)
        print(f"\r{UI.progress_bar(len(items), len(items))}")  # 完成進度條
        self._pretty_print_json(result, "批量處理結果")
        return result
    
    def batch_process_sync(self, items: List[str], operation: str, model: str = "GPT-4.1") -> Dict:
        """同步版本的批量處理"""
        return asyncio.run(self.batch_process(items, operation, model))

# ========================================
# 使用範例和測試函數
# ========================================

def demo_all_functions():
    """演示所有功能的美化效果"""
    print(UI.header(" 🎨 Enhanced AskBot 美化演示 "))
    
    # 請將 'your-api-key' 替換為實際的 API Key
    api_key = API_KEY
    try:
        bot = AskBot(api_key)
    except Exception as e:
        print(UI.error(f"初始化失敗: {e}"))
        return
    
    print(UI.info("開始演示所有功能的美化效果..."))
    
    # 1. 演示選擇功能
    print(f"\n{UI.section('1. 智能選擇功能演示')}")
    bot.choice_sync(
        options=["Python", "JavaScript", "Go", "Rust"],
        criteria="最適合AI開發的程式語言"
    )
    
    # 2. 演示列舉功能
    print(f"\n{UI.section('2. 智能列舉功能演示')}")
    bot.enumerate_sync("機器學習的應用領域", 3)
    
    # 3. 演示建議功能
    print(f"\n{UI.section('3. 專業建議功能演示')}")
    bot.suggest_sync(
        situation="我是程式設計新手，想學習AI開發",
        goal="3個月內掌握基礎AI技能"
    )
    
    print(UI.success("演示完成！"))

def interactive_mode():
    """互動模式"""
    print(UI.header(" 🤖 Enhanced AskBot 互動模式 "))
    
    api_key =API_KEY
    
    try:
        bot = AskBot(api_key)
    except Exception as e:
        print(UI.error(f"初始化失敗: {e}"))
        return
    
    print(UI.info("可用功能："))
    functions = [
        "ask - 基礎問答",
        "choice - 智能選擇",
        "enumerate - 智能列舉", 
        "suggest - 專業建議",
        "translate - 翻譯",
        "summarize - 總結",
        "plan - 規劃",
        "batch - 批量處理"
    ]
    
    for func in functions:
        print(f"  • {UI.code(func)}")
    
    print(f"\n{UI.warning('輸入 demo 運行演示，quit 退出')}")
    print(UI.separator())
    
    while True:
        try:
            command = input(f"\n{UI.question('輸入命令或問題')}: ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print(UI.success("再見！"))
                break
            
            if command.lower() == 'demo':
                demo_all_functions()
                continue
            
            if not command:
                continue
            
            print(UI.processing("AI 正在思考..."))
            answer = bot.ask_sync(command)
            
            print(f"\n{UI.separator()}")
            print(UI.answer("AI 回答"))
            print(UI.separator())
            print(answer)
            
        except KeyboardInterrupt:
            print(f"\n{UI.success('再見！')}")
            break
        except Exception as e:
            print(UI.error(f"執行錯誤: {e}"))

def main():
    """主程序"""
    # 啟用 ANSI 支援 (Windows)
    if os.name == 'nt':
        try:
            os.system('')
        except:
            pass
    
    print(UI.header(" 🚀 Enhanced AskBot - 美化版 "))
    print(UI.info("支援 ANSI 色彩的增強版 AI 對話工具"))
    print(UI.separator())
    
    interactive_mode()

if __name__ == "__main__":
    main()
