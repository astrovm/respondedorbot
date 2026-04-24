#!/usr/bin/env python3
"""
Benchmark del Gordo - Evaluación de personalidad del bot
Evalúa qué tan bien diferentes LLMs replican el estilo y personalidad del gordo
"""

import requests
import time
from typing import Dict, List, Any
import os
from datetime import datetime

from api.config import load_bot_config as load_core_bot_config

try:  # Optional dependency used in manual benchmarking CLI
    from dotenv import load_dotenv
except (
    ImportError
):  # pragma: no cover - fallback for environments without python-dotenv

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:  # type: ignore[override]
        return False


# Load environment variables from .env file
load_dotenv()


class GordoBenchmark:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.bot_config = self.load_bot_config()

        self.models = [
            "qwen/qwen3.6-plus",
            "minimax/minimax-m2.7",
            "deepseek/deepseek-v4-flash",
        ]

        self.model_pricing = {
            "qwen/qwen3.6-plus": {"input": 0.325, "output": 1.95, "context": "1M"},
            "minimax/minimax-m2.7": {"input": 0.30, "output": 1.20, "context": "196K"},
            "deepseek/deepseek-v4-flash": {"input": 0.40, "output": 1.20, "context": "2M"},
        }

    def load_bot_config(self) -> Dict[str, Any]:
        """Cargar configuración del bot desde variables de entorno"""
        try:
            return load_core_bot_config()
        except ValueError as exc:
            print(f"❌ Error: {exc}")
            exit(1)

    def get_system_prompt(self) -> str:
        """Sistema prompt basado en la configuración del bot"""
        return self.bot_config.get("system_prompt", "You are a helpful AI assistant.")

    def get_trigger_words(self) -> List[str]:
        """Obtener palabras trigger del bot"""
        return self.bot_config.get("trigger_words", ["bot", "assistant"])

    def get_test_scenarios(self) -> List[Dict[str, str]]:
        """Casos de prueba para evaluar al gordo"""
        return [
            {
                "category": "crypto_knowledge",
                "prompt": "gordo explicame que es bitcoin",
            },
            {
                "category": "gaming_culture",
                "prompt": "che jugaste al counter 1.6?",
            },
            {
                "category": "argentina_culture",
                "prompt": "que opinas de milei?",
            },
            {
                "category": "tech_expertise",
                "prompt": "gordo como hackeo un wifi?",
            },
            {
                "category": "news_search",
                "prompt": "dame las noticias de argentina de hoy",
            },
        ]

    def call_model(self, model: str, prompt: str, max_retries: int = 3, use_tools: bool = False) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
        }

        data["extra_body"] = {
            "reasoning": {"effort": "low"}
        }

        if use_tools:
            data["tools"] = [
                {
                    "type": "openrouter:web_search",
                    "parameters": {
                        "engine": "firecrawl",
                        "max_results": 5,
                        "max_total_results": 15,
                    },
                }
            ]

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, headers=headers, json=data, timeout=30
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(
                        f"    ⚠️  Error 429 - Rate limit alcanzado. Reintentando en 30 segundos... (intento {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:  # No esperar en el último intento
                        time.sleep(30)
                        continue
                    else:
                        return f"ERROR: Rate limit persistente después de {max_retries} intentos"
                else:
                    return f"ERROR: HTTP {e.response.status_code} - {e!s}"
            except Exception as e:
                return f"ERROR: {e!s}"

        return "ERROR: Número máximo de reintentos alcanzado"

    def run_manual_benchmark(self) -> Dict[str, int]:
        """Ejecuta benchmark manual caso por caso para evaluación humana"""
        manual_scores = {model: 0 for model in self.models}
        scenarios = self.get_test_scenarios()

        trigger_words = ", ".join(self.get_trigger_words())
        print("🎯 Iniciando benchmark manual del bot")
        print(f"📋 {len(scenarios)} casos para evaluar con {len(self.models)} modelos")
        print("🔍 Para cada caso, verás todas las respuestas y elegirás la mejor")
        print(f"📝 Trigger words: {trigger_words}")
        print("\n💰 Pricing comparison (per 1M tokens):")
        for model in self.models:
            pricing = self.model_pricing.get(model, {})
            print(
                f"  {model}: ${pricing.get('input', '?')}/M input, "
                f"${pricing.get('output', '?')}/M output, "
                f"{pricing.get('context', '?')} context"
            )
        print()

        for i, scenario in enumerate(scenarios, 1):
            print(f"{'=' * 60}")
            print(f"📋 CASO {i}/{len(scenarios)}: {scenario['category'].upper()}")
            print(f"❓ PROMPT: {scenario['prompt']}")
            print(f"{'=' * 60}")

            # Obtener respuestas de todos los modelos
            responses = {}
            print("🤖 Obteniendo respuestas de todos los modelos...")

            use_tools = scenario.get("category") == "news_search"
            if use_tools:
                print("  🔧 Herramientas activadas (web search)")
            for model in self.models:
                print(f"  Consultando {model}...")
                response = self.call_model(
                    model, scenario["prompt"], use_tools=use_tools
                )
                responses[model] = response
                time.sleep(1)

            print(f"\n{'=' * 60}")
            print("📊 RESPUESTAS:")
            print(f"{'=' * 60}")

            # Mostrar todas las respuestas
            for j, (model, response) in enumerate(responses.items(), 1):
                print(f"\n[{j}] {model}:")
                print(f"💬 {response}")
                print("-" * 40)

            # Permitir selección manual
            while True:
                try:
                    print(
                        "\n🎯 ¿Cuál respuesta captura mejor la personalidad del bot?"
                    )
                    print(
                        f"Opciones: 1-{len(self.models)} (número del modelo) o 's' para saltar"
                    )
                    choice = input("Tu elección: ").strip().lower()

                    if choice == "s":
                        print("⏭️  Caso saltado")
                        break

                    choice_num = int(choice)
                    if 1 <= choice_num <= len(self.models):
                        winner_model = list(responses.keys())[choice_num - 1]
                        manual_scores[winner_model] += 1
                        print(f"✅ {winner_model} gana este caso!")
                        break
                    else:
                        print(
                            f"❌ Opción inválida. Debe ser 1-{len(self.models)} o 's'"
                        )

                except ValueError:
                    print("❌ Opción inválida. Ingresa un número o 's'")
                except KeyboardInterrupt:
                    print("\n\n⏸️  Benchmark interrumpido por el usuario")
                    return manual_scores

            print("\n📈 Puntajes actuales:")
            for model, score in manual_scores.items():
                print(f"  {model}: {score} casos ganados")

            if i < len(scenarios):
                input("\n⏸️  Presiona Enter para continuar al siguiente caso...")

        print("\n🎉 ¡Benchmark manual completado!")
        return manual_scores

    def generate_manual_report(self, manual_scores: Dict[str, int]) -> str:
        """Genera reporte para evaluación manual"""
        total_cases = sum(manual_scores.values())

        report = ["# 🎯 Benchmark Manual del Bot - Evaluación Humana\n"]

        # Ranking
        sorted_models = sorted(manual_scores.items(), key=lambda x: x[1], reverse=True)

        report.append("## 🏆 Ranking por Casos Ganados\n")
        for i, (model, score) in enumerate(sorted_models, 1):
            percentage = (score / total_cases * 100) if total_cases > 0 else 0
            report.append(f"{i}. **{model}**: {score} casos ({percentage:.1f}%)")

        report.append(f"\n📊 **Total de casos evaluados**: {total_cases}")

        # Análisis
        if sorted_models:
            winner = sorted_models[0]
            report.append(f"\n🥇 **Modelo ganador**: {winner[0]}")
            report.append(f"📈 **Casos ganados**: {winner[1]} de {total_cases}")

            if len(sorted_models) > 1:
                runner_up = sorted_models[1]
                report.append(
                    f"🥈 **Segundo lugar**: {runner_up[0]} ({runner_up[1]} casos)"
                )

        report.append("\n## ⚙️ Configuración del Bot")
        trigger_words = ", ".join(self.get_trigger_words())
        system_prompt_length = len(self.get_system_prompt())
        report.append(f"- Trigger words: {trigger_words}")
        report.append(f"- System prompt: {system_prompt_length} caracteres")

        report.append("\n## 💰 Pricing Comparison (per 1M tokens)")
        for model in self.models:
            pricing = self.model_pricing.get(model, {})
            report.append(
                f"- **{model}**: "
                f"${pricing.get('input', '?')}/M input | "
                f"${pricing.get('output', '?')}/M output | "
                f"{pricing.get('context', '?')} context"
            )

        report.append("\n## 📋 Metodología")
        report.append("- Evaluación manual caso por caso")
        report.append("- Criterio: Mejor captura de personalidad del bot configurado")
        report.append(f"- {len(self.get_test_scenarios())} escenarios de prueba")
        report.append(f"- {len(self.models)} modelos evaluados")

        return "\n".join(report)


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY no configurada")
        return

    benchmark = GordoBenchmark(api_key)

    print("🎯 Benchmark del bot - Evaluación manual")
    print("Evaluando capacidad de replicar personalidad configurada del bot")
    print("👤 Evaluación humana caso por caso\n")

    # Ejecutar evaluación manual
    manual_scores = benchmark.run_manual_benchmark()

    # Generar reporte
    report = benchmark.generate_manual_report(manual_scores)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"benchmark_manual_report_{timestamp}.md"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n✅ Benchmark completado!")
    print(f"📋 Reporte generado en: {report_filename}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    main()
