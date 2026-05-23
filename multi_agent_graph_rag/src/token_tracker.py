from dataclasses import dataclass, field


@dataclass
class ModelCall:
    pipeline: str
    stage: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        prices = {
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o":      (5.00, 15.00),
        }
        in_p, out_p = prices.get(self.model, (0.15, 0.60))
        return (self.input_tokens * in_p + self.output_tokens * out_p) / 1_000_000


@dataclass
class TokenTracker:
    calls: list[ModelCall] = field(default_factory=list)

    def record(self, pipeline: str, stage: str, model: str, usage) -> ModelCall:
        call = ModelCall(
            pipeline=pipeline,
            stage=stage,
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0),
        )
        self.calls.append(call)
        return call

    def summary_by_pipeline(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for call in self.calls:
            p = result.setdefault(call.pipeline, {
                "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "cost_usd": 0.0, "api_calls": 0,
            })
            p["input_tokens"]+= call.input_tokens
            p["output_tokens"]+= call.output_tokens
            p["total_tokens"]+= call.total_tokens
            p["cost_usd"] += call.cost_usd
            p["api_calls"]+= 1
        return result

    def print_report(self) -> None:
        summary = self.summary_by_pipeline()
        print("Token usage comparison usage")
        print("-" * 65)
        print(f"{'Pipeline':<30} {'Calls':>5} {'Input':>8} {'Output':>8} {'Total':>8} {'Cost $':>8}")
        print("-" * 65)
        for name, s in summary.items():
            print(f"{name:<30} {s['api_calls']:>5} {s['input_tokens']:>8,} "
                  f"{s['output_tokens']:>8,} {s['total_tokens']:>8,} {s['cost_usd']:>8.4f}")
        print("-" * 65)

        if len(summary) > 1:
            peak = max(s["total_tokens"] for s in summary.values())
            print("\n  Savings vs most expensive pipeline:")
            for name, s in summary.items():
                saved = peak - s["total_tokens"]
                pct = saved / peak * 100 if peak else 0
                print(f"  {name:<30} saves {saved:>8,} tokens ({pct:.1f}%)")
        print()
