"""Model pricing tables and cost calculation for paid API models."""

# Paid model rates per 1,000 tokens (USD)
# pony: code (Flash 3.8) $0.50/MTok input, $1.50/MTok output. Jev $0.042/MTok input.
PAID_MODEL_PRICING: dict[str, dict[str, float]] = {
    "code": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "flash": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "gemini-3.8-flash": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    "typesafe/jev-1.13": {"input_per_1k": 0.000042, "output_per_1k": 0.0},
}


def calculate_paid_cost(model_usage: dict[str, dict[str, int]]) -> float | None:
    """Return USD cost from paid models only, or None if no paid model was used."""
    total = 0.0
    found_paid = False
    for model_name, counts in model_usage.items():
        pricing = PAID_MODEL_PRICING.get(model_name)
        if pricing:
            found_paid = True
            total += (counts.get("input_tokens", 0) / 1000.0) * pricing["input_per_1k"]
            total += (counts.get("output_tokens", 0) / 1000.0) * pricing["output_per_1k"]
    return round(total, 6) if found_paid else None
