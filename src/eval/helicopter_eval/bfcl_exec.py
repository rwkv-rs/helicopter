from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .apibank import decode_tool_calls
from .bfcl_ast import _coerce_list, _normalize_tool_schema, _read_items, _render_bfcl_question
from .openai_client import chat_completion
from .sampling import apply_limit_or_sample, dataset_sample_suffix
from .scoreboard import ScoreboardEvalResult, ScoreboardWriteConfig, write_scoreboard_results


BFCL_EXEC_CATEGORY_PATHS: dict[str, tuple[str, str]] = {
    "exec_simple": (
        "unused_datasets/question/BFCL_v4_exec_simple.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_simple.json",
    ),
    "exec_multiple": (
        "unused_datasets/question/BFCL_v4_exec_multiple.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_multiple.json",
    ),
    "exec_parallel": (
        "unused_datasets/question/BFCL_v4_exec_parallel.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_parallel.json",
    ),
    "exec_parallel_multiple": (
        "unused_datasets/question/BFCL_v4_exec_parallel_multiple.json",
        "unused_datasets/possible_answer/BFCL_v4_exec_parallel_multiple.json",
    ),
}


@dataclass(frozen=True, slots=True)
class BfclExecCallResult:
    call: str
    success: bool
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BfclExecSample:
    sample_index: int
    task_id: str
    instruction: str
    tools: tuple[dict[str, Any], ...]
    expected_executable_calls: tuple[str, ...]
    execution_result_type: tuple[str, ...]
    category: str


@dataclass(frozen=True, slots=True)
class BfclExecResult:
    sample_index: int
    task_id: str
    prompt: str
    completion: str
    answer: str
    reference_answer: str
    is_passed: bool
    fail_reason: str

    def to_scoreboard(self) -> ScoreboardEvalResult:
        return ScoreboardEvalResult(
            sample_index=self.sample_index,
            prompt=self.prompt,
            completion=self.completion,
            answer=self.answer,
            reference_answer=self.reference_answer,
            is_passed=self.is_passed,
            fail_reason=self.fail_reason,
        )


@dataclass(frozen=True, slots=True)
class BfclExecRunConfig:
    base_url: str
    model: str
    benchmark: str
    category: str
    limit: int | None = None
    sample_size: int | None = None
    sample_seed: int = 42
    split: str = "test"
    source_root: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768
    timeout_s: float = 600.0
    scoreboard_dataset: str | None = None
    job_name: str = "function_bfcl_exec"
    job_id: str | None = None
    runner: str = "helicopter_eval.bfcl_exec"
    cot_mode: str = "CoT"


class BfclExecSandbox:
    def __init__(self) -> None:
        self._functions: dict[str, Any] = {
            "add_binary_numbers": lambda a, b: bin(int(str(a), 2) + int(str(b), 2))[2:],
            "adjust_for_inflation": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "apply_discount": lambda price, discount: price * (1 - discount),
            "book_room": lambda **kwargs: {"booking_confirmed": True, **kwargs},
            "calc_binomial_probability": self._calc_binomial_probability,
            "calculate_basal_metabolic_rate": self._calculate_basal_metabolic_rate,
            "calculate_cosine_similarity": self._calculate_cosine_similarity,
            "calculate_daily_energy_expenditure": lambda bmr, activity_factor: bmr * activity_factor,
            "calculate_density": lambda mass, volume: mass / volume,
            "calculate_displacement": lambda initial_velocity, acceleration, time: (
                initial_velocity * time + 0.5 * acceleration * time**2
            ),
            "calculate_electrostatic_potential_energy": lambda charge, voltage: charge * voltage,
            "calculate_final_velocity": lambda initial_velocity, acceleration, time: (
                initial_velocity + acceleration * time
            ),
            "calculate_future_value": lambda present_value, interest_rate, periods: (
                present_value * ((1 + interest_rate) ** periods)
            ),
            "calculate_intercept": lambda x1, y1, slope: y1 - slope * x1,
            "calculate_interest_rate": lambda principal, amount, time: (amount / principal) ** (1 / time) - 1,
            "calculate_investment_value": self._calculate_investment_value,
            "calculate_mean": lambda numbers: sum(numbers) / len(numbers),
            "calculate_nutritional_needs": self._calculate_nutritional_needs,
            "calculate_permutations": lambda n, k: math.factorial(int(n)) // math.factorial(int(n) - int(k)),
            "calculate_slope": lambda x1, y1, x2, y2: (y2 - y1) / (x2 - x1),
            "calculate_standard_deviation": self._calculate_standard_deviation,
            "calculate_total": lambda price, quantity: price * quantity,
            "calculate_total_price": lambda price, quantity: price * quantity,
            "calculate_triangle_area": lambda base, height: 0.5 * base * height,
            "compound_interest": lambda principal, rate, time, n=1: principal * ((1 + rate / n) ** (n * time)),
            "confirm_booking": lambda **kwargs: {"confirmed": True, **kwargs},
            "convert_binary_to_decimal": lambda binary: int(str(binary), 2),
            "convert_coordinates": lambda latitude, longitude: {"latitude": latitude, "longitude": longitude},
            "convert_decimal_to_hex": lambda decimal: hex(int(decimal)),
            "convert_temperature": self._convert_temperature,
            "estimate_derivative": self._estimate_derivative,
            "generate_random_number": lambda min_value=0, max_value=100: int((int(min_value) + int(max_value)) / 2),
            "geometry_area_circle": lambda radius: math.pi * radius * radius,
            "get_distance": lambda pointA, pointB: math.dist(pointA, pointB),
            "get_fibonacci_number": self._get_fibonacci_number,
            "get_fibonacci_sequence": self._get_fibonacci_sequence,
            "get_prime_factors": self._get_prime_factors,
            "inflation_adjustment": lambda amount, inflation_rate, years: amount * ((1 + inflation_rate) ** years),
            "linear_regression": self._linear_regression,
            "mat_mul": self._mat_mul,
            "math_factorial": lambda n: math.factorial(int(n)),
            "math_gcd": lambda a, b: math.gcd(int(a), int(b)),
            "math_lcm": lambda a, b: abs(int(a) * int(b)) // math.gcd(int(a), int(b)),
            "maxPoints": self._max_points,
            "mortgage_calculator": self._mortgage_calculator,
            "order_food": self._order_food,
            "polygon_area": self._polygon_area,
            "predict_value": lambda slope, intercept, x: slope * x + intercept,
            "quadratic_roots": self._quadratic_roots,
            "sort_array": lambda array, reverse=False: sorted(array, reverse=bool(reverse)),
            "validate_polygon": lambda vertices: len(vertices) >= 3,
        }

    def execute(self, call: str) -> BfclExecCallResult:
        try:
            name, args, kwargs = _parse_call(call)
            func = self._functions.get(name)
            result = func(*args, **kwargs) if func is not None else self._execute_fixture(name, args, kwargs)
            return BfclExecCallResult(call=call, success=True, result=_json_safe(result))
        except Exception as exc:  # noqa: BLE001
            return BfclExecCallResult(call=call, success=False, error=str(exc))

    def _execute_fixture(self, name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name in {"get_stock_price_by_stock_name", "get_company_name_by_stock_name", "get_stock_history"}:
            stock = str(kwargs.get("stock_name") or (args[0] if args else "")).upper()
            if name == "get_company_name_by_stock_name":
                companies = {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc."}
                return companies.get(stock, stock)
            if name == "get_stock_history":
                return {"symbol": stock, "history": [{"close": self._stock_price(stock)}]}
            return self._stock_price(stock)
        if name == "get_weather_data":
            coordinates = kwargs.get("coordinates") or (args[0] if args else [0, 0])
            if isinstance(coordinates, Mapping):
                latitude = coordinates.get("latitude", coordinates.get("lat", 0))
                longitude = coordinates.get("longitude", coordinates.get("lon", coordinates.get("long", 0)))
            else:
                latitude = coordinates[0] if len(coordinates) > 0 else 0
                longitude = coordinates[1] if len(coordinates) > 1 else 0
            return {"temperature": round(float(latitude) * 0.1 + float(longitude) * 0.01, 3), "unit": "celsius"}
        if name in {"get_coordinate_by_ip_address", "get_zipcode_by_ip_address"}:
            ip_address = str(kwargs.get("ip_address") or (args[0] if args else ""))
            return "private range" if ip_address.startswith("192.168.") else {"ip_address": ip_address}
        if name in {"get_coordinates_from_city", "retrieve_city_based_on_zipcode", "get_time_zone_by_coord"}:
            return self._location_fixture(name, args, kwargs)
        if name in {"get_covid_death_by_country", "get_active_covid_case_by_country"}:
            country = str(kwargs.get("country") or (args[0] if args else "")).lower()
            base = sum(ord(char) for char in country)
            return base * (1000 if "death" in name else 500)
        if name in {"get_rating_by_amazon_ASIN", "get_price_by_amazon_ASIN", "get_product_name_by_amazon_ASIN"}:
            asin = str(kwargs.get("ASIN") or kwargs.get("asin") or (args[0] if args else ""))
            if "rating" in name:
                return str(round(3.5 + (sum(ord(c) for c in asin) % 15) / 10, 1))
            if "price" in name:
                return f"${50 + (sum(ord(c) for c in asin) % 500)}.00"
            return f"Product {asin}"
        if name in {"get_movie_director", "get_director_by_movie_name", "get_movie_rating", "get_movie_genre"}:
            movie = str(kwargs.get("movie_name") or (args[0] if args else "")).lower()
            movies = {
                "avatar": {"director": "James Cameron", "rating": "PG-13", "genre": "Science Fiction"},
                "pulp fiction": {"director": "Quentin Tarantino", "rating": "R", "genre": "Crime"},
            }
            info = movies.get(movie, {"director": "Unknown", "rating": "Unknown", "genre": "Unknown"})
            if "rating" in name:
                return info["rating"]
            if "genre" in name:
                return info["genre"]
            return info["director"]
        if name == "retrieve_holiday_by_year":
            country = kwargs.get("country") or (args[0] if args else "")
            year = int(kwargs.get("year") or (args[1] if len(args) > 1 else 2023))
            return [
                {
                    "countryCode": str(country),
                    "date": f"{year:04d}-01-01",
                    "localName": "New Year",
                    "name": "New Year's Day",
                }
            ]
        if name == "find_term_on_urban_dictionary":
            term = str(kwargs.get("term") or (args[0] if args else ""))
            return {"term": term, "definition": f"Definition for {term}"}
        if name == "convert_currency":
            amount = float(kwargs.get("amount") or (args[0] if args else 0.0))
            from_currency = str(kwargs.get("from_currency") or (args[1] if len(args) > 1 else "")).upper()
            to_currency = str(kwargs.get("to_currency") or (args[2] if len(args) > 2 else "")).upper()
            rates = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.08, ("USD", "GBP"): 0.79, ("GBP", "USD"): 1.27}
            return amount * rates.get((from_currency, to_currency), 1.0)
        raise ValueError(f"unsupported official BFCL executable function: {name}")

    @staticmethod
    def _stock_price(stock: str) -> float:
        return {"AAPL": 169.02, "MSFT": 421.9, "GOOG": 175.4, "META": 477.2, "NFLX": 610.1, "BABA": 75.0}.get(
            stock,
            100.0,
        )

    @staticmethod
    def _location_fixture(name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        if name == "retrieve_city_based_on_zipcode":
            return {"90210": "BEVERLY HILLS", "10001": "NEW YORK", "08540": "PRINCETON"}.get(
                str(kwargs.get("zipcode") or ""),
                "UNKNOWN",
            )
        if name == "get_coordinates_from_city":
            city = str(kwargs.get("city_name") or (args[0] if args else ""))
            return {"city": city, "latitude": "0.0", "longitude": "0.0"}
        return "UTC"

    @staticmethod
    def _calc_binomial_probability(n: int, k: int, p: float) -> float:
        return math.comb(int(n), int(k)) * (float(p) ** int(k)) * ((1 - float(p)) ** (int(n) - int(k)))

    @staticmethod
    def _calculate_cosine_similarity(vectorA: Sequence[float], vectorB: Sequence[float]) -> float:
        dot = sum(float(a) * float(b) for a, b in zip(vectorA, vectorB))
        norm_a = math.sqrt(sum(float(a) ** 2 for a in vectorA))
        norm_b = math.sqrt(sum(float(b) ** 2 for b in vectorB))
        return dot / (norm_a * norm_b)

    @staticmethod
    def _calculate_standard_deviation(numbers: Sequence[float]) -> float:
        mean = sum(numbers) / len(numbers)
        return math.sqrt(sum((float(item) - mean) ** 2 for item in numbers) / len(numbers))

    @staticmethod
    def _calculate_basal_metabolic_rate(weight: float, height: float, age: int, gender: str) -> float:
        offset = 5 if str(gender).lower() == "male" else -161
        return 10 * weight + 6.25 * height - 5 * age + offset

    def _calculate_nutritional_needs(
        self,
        weight: float,
        height: float,
        age: int,
        gender: str,
        activity_level: float,
        goal: str,
    ) -> dict[str, float]:
        bmr = self._calculate_basal_metabolic_rate(weight, height, age, gender)
        calories = bmr * float(activity_level)
        if str(goal).lower().startswith("lose"):
            calories -= 500
        elif str(goal).lower().startswith("gain"):
            calories += 500
        return {"calories": calories}

    @staticmethod
    def _calculate_investment_value(
        initial_investment: float,
        annual_contribution: float,
        years: int,
        annual_return: float,
        inflation_rate: Sequence[float] | float = (),
        adjust_for_inflation: bool = True,
    ) -> float:
        value = float(initial_investment)
        if isinstance(inflation_rate, (int, float)):
            rates: Sequence[float] = [float(inflation_rate)]
        elif inflation_rate is None:
            rates = []
        else:
            rates = inflation_rate
        for index in range(int(years)):
            inflation = float(rates[index]) if adjust_for_inflation and index < len(rates) else 0.0
            value = (value + float(annual_contribution)) * (1 + float(annual_return) - inflation)
        return value

    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        source = str(from_unit).lower()
        target = str(to_unit).lower()
        celsius = (value - 32) * 5 / 9 if source.startswith("f") else value
        return celsius * 9 / 5 + 32 if target.startswith("f") else celsius

    @staticmethod
    def _estimate_derivative(function: str, x: float) -> float:
        source = str(function).strip()
        if not source.startswith("lambda"):
            source = f"lambda x: {source}"
        fn = eval(source, {"__builtins__": {}, "math": math}, {})  # noqa: S307 - restricted BFCL math expression.
        h = 1e-5
        return (fn(x + h) - fn(x - h)) / (2 * h)

    @staticmethod
    def _get_fibonacci_number(n: int) -> int:
        seq = BfclExecSandbox._get_fibonacci_sequence(int(n) + 1)
        return seq[-1] if seq else 0

    @staticmethod
    def _get_fibonacci_sequence(n: int) -> list[int]:
        values = [0, 1]
        while len(values) < int(n):
            values.append(values[-1] + values[-2])
        return values[: int(n)]

    @staticmethod
    def _get_prime_factors(
        n: int | None = None,
        *,
        number: int | None = None,
        formatted: bool | str | None = None,
    ) -> Any:
        factors: list[int] = []
        value = int(number if number is not None else n)
        divisor = 2
        while divisor * divisor <= value:
            while value % divisor == 0:
                factors.append(divisor)
                value //= divisor
            divisor += 1
        if value > 1:
            factors.append(value)
        if _truthy_exec_flag(formatted):
            return " * ".join(str(item) for item in factors)
        return factors

    @staticmethod
    def _order_food(
        item: Sequence[str] | None = None,
        quantity: Sequence[float] | None = None,
        price: Sequence[float] | None = None,
        *,
        items: Sequence[str] | None = None,
        **_kwargs: Any,
    ) -> float:
        selected_items = item if item is not None else items
        quantities = quantity or [1 for _ in (selected_items or [])]
        prices = price or [0 for _ in (selected_items or [])]
        return sum(float(q) * float(p) for q, p in zip(quantities, prices))

    @staticmethod
    def _linear_regression(x: Sequence[float], y: Sequence[float], point: float) -> float:
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        denom = sum((item - mean_x) ** 2 for item in x)
        slope = 0.0 if denom == 0 else sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denom
        return slope * point + (mean_y - slope * mean_x)

    @staticmethod
    def _mat_mul(matA: Sequence[Sequence[float]], matB: Sequence[Sequence[float]]) -> list[list[float]]:
        columns = list(zip(*matB))
        return [[sum(a * b for a, b in zip(row, col)) for col in columns] for row in matA]

    @staticmethod
    def _max_points(points: Sequence[Sequence[float]]) -> int:
        if len(points) <= 2:
            return len(points)
        best = 2
        for i, a in enumerate(points):
            slopes: dict[tuple[float, float], int] = {}
            for b in points[i + 1 :]:
                dx = float(b[0]) - float(a[0])
                dy = float(b[1]) - float(a[1])
                key = (0.0, 1.0) if dx == 0 else (1.0, round(dy / dx, 12))
                slopes[key] = slopes.get(key, 1) + 1
                best = max(best, slopes[key])
        return best

    @staticmethod
    def _mortgage_calculator(loan_amount: float, interest_rate: float, loan_period: int) -> float:
        monthly_rate = float(interest_rate) / 12
        payments = int(loan_period) * 12
        if monthly_rate == 0:
            return loan_amount / payments
        return loan_amount * monthly_rate * ((1 + monthly_rate) ** payments) / (((1 + monthly_rate) ** payments) - 1)

    @staticmethod
    def _polygon_area(vertices: Sequence[Sequence[float]]) -> float:
        area = 0.0
        for index, point in enumerate(vertices):
            next_point = vertices[(index + 1) % len(vertices)]
            area += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
        return abs(area) / 2

    @staticmethod
    def _quadratic_roots(a: float, b: float, c: float) -> list[float | str]:
        disc = b * b - 4 * a * c
        if disc < 0:
            return ["complex"]
        root = math.sqrt(disc)
        return [(-b + root) / (2 * a), (-b - root) / (2 * a)]


def load_samples(config: BfclExecRunConfig) -> list[BfclExecSample]:
    if config.split != "test":
        raise ValueError("BFCL v4 executable datasets only provide test split")
    if config.category not in BFCL_EXEC_CATEGORY_PATHS:
        raise ValueError(f"unknown BFCL exec category: {config.category}")
    question_rel, answer_rel = BFCL_EXEC_CATEGORY_PATHS[config.category]
    questions = _read_items(config, question_rel)
    answers = {
        str(item.get("id") or item.get("task_id") or ""): item
        for item in _read_items(config, answer_rel)
        if isinstance(item, Mapping)
    }
    samples: list[BfclExecSample] = []
    for index, item in enumerate(questions):
        if config.limit is not None and config.sample_size is None and len(samples) >= int(config.limit):
            break
        if not isinstance(item, Mapping):
            continue
        task_id = str(item.get("id") or item.get("task_id") or f"{config.category}_{index}")
        answer = answers.get(task_id)
        if answer is None:
            raise ValueError(f"missing BFCL executable possible-answer entry for {task_id}")
        instruction = _render_bfcl_question(item.get("question"))
        expected_calls = _ground_truth_to_exec_calls(answer.get("ground_truth"))
        execution_types = [
            str(value).strip() or "exact_match"
            for value in _coerce_list(answer.get("execution_result_type"))
        ]
        samples.append(
            BfclExecSample(
                sample_index=len(samples),
                task_id=task_id,
                instruction=instruction,
                tools=tuple(_normalize_tool_schema(tool) for tool in _coerce_list(item.get("function"))),
                expected_executable_calls=tuple(expected_calls),
                execution_result_type=tuple(execution_types or ["exact_match"] * len(expected_calls)),
                category=config.category,
            )
        )
    return apply_limit_or_sample(
        samples,
        limit=config.limit,
        sample_size=config.sample_size,
        sample_seed=config.sample_seed,
        sort_key=lambda sample: sample.sample_index,
    )


def build_prompt(sample: BfclExecSample) -> str:
    tools = json.dumps(sample.tools, ensure_ascii=False, indent=2, sort_keys=True)
    schema = json.dumps(
        {
            "oneOf": [
                {
                    "type": "object",
                    "required": ["name", "arguments"],
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                },
                {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "arguments"],
                        "additionalProperties": False,
                        "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                    },
                    "minItems": 1,
                },
            ]
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = (
        "You are solving a Berkeley Function Calling Leaderboard executable task.\n\n"
        "Tools:\n"
        f"{tools}\n\n"
        "Output JSON schema:\n"
        f"{schema}\n\n"
        "Return exactly one JSON value that validates against the schema. "
        "For one required call, return one JSON object. "
        "For multiple required calls, return a JSON array containing every required call. "
        "Use only listed tool names. Return no prose, no markdown, and no extra text outside the JSON value.\n\n"
        f"User request:\n{sample.instruction}\n\n"
        "Tool call:"
    )
    if _force_array_prefix(sample):
        prompt += "\n["
    return prompt


def render_bfcl_exec_call(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or call.get("tool_name") or "").strip()
    arguments = call.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = {}
    rendered = ", ".join(f"{key}={_python_literal(value)}" for key, value in arguments.items())
    return f"{name}({rendered})" if rendered else f"{name}()"


def evaluate_completion(
    sample: BfclExecSample,
    completion: str,
    *,
    sandbox: BfclExecSandbox | None = None,
) -> tuple[bool, str, list[dict[str, Any]], dict[str, Any]]:
    sandbox = sandbox or BfclExecSandbox()
    decision_completion = _complete_forced_prefix(sample, completion)
    try:
        decoded_calls = decode_tool_calls(decision_completion)
    except Exception as exc:  # noqa: BLE001
        return False, f"parse_error:{exc}", [], {"parse_error": str(exc)}
    evaluation = evaluate_bfcl_exec_calls(sample, decoded_calls, sandbox=sandbox)
    return evaluation[0], evaluation[1], decoded_calls, evaluation[2]


def evaluate_bfcl_exec_calls(
    sample: BfclExecSample,
    decoded_calls: Sequence[Mapping[str, Any]],
    *,
    sandbox: BfclExecSandbox | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    sandbox = sandbox or BfclExecSandbox()
    expected_calls = list(sample.expected_executable_calls)
    model_calls = [render_bfcl_exec_call(call) for call in decoded_calls]
    match_types = list(sample.execution_result_type) or ["exact_match"] * len(expected_calls)
    while len(match_types) < len(expected_calls):
        match_types.append("exact_match")

    expected_results = [sandbox.execute(call) for call in expected_calls]
    model_results = [sandbox.execute(call) for call in model_calls]
    details: dict[str, Any] = {
        "expected_executable_calls": expected_calls,
        "decoded_executable_calls": model_calls,
        "execution_result_type": match_types,
        "expected_execution_results": [_result_payload(item) for item in expected_results],
        "decoded_execution_results": [_result_payload(item) for item in model_results],
    }
    failure_bits: list[str] = []
    passed_count = 0

    if _is_parallel_record(sample):
        matched_model_indices: set[int] = set()
        for expected_index, expected in enumerate(expected_results):
            match_type = match_types[expected_index] if expected_index < len(match_types) else "exact_match"
            for model_index, actual in enumerate(model_results):
                if model_index in matched_model_indices:
                    continue
                ok, reason = _execution_result_matches(actual, expected, match_type)
                if ok:
                    matched_model_indices.add(model_index)
                    passed_count += 1
                    break
            else:
                failure_bits.append(f"call_{expected_index}:missing_or_mismatch")
        for model_index in range(len(model_results)):
            if model_index not in matched_model_indices:
                failure_bits.append(f"call_{model_index}:unexpected_extra_call")
        is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
        return bool(is_passed), "; ".join(failure_bits), details

    max_len = max(len(expected_results), len(model_results))
    for index in range(max_len):
        if index >= len(expected_results):
            failure_bits.append(f"call_{index}:unexpected_extra_call")
            continue
        if index >= len(model_results):
            failure_bits.append(f"call_{index}:missing_call")
            continue
        ok, reason = _execution_result_matches(model_results[index], expected_results[index], match_types[index])
        if ok:
            passed_count += 1
        else:
            failure_bits.append(f"call_{index}:{reason}")
    is_passed = len(model_results) == len(expected_results) and passed_count == len(expected_results)
    return bool(is_passed), "; ".join(failure_bits), details


def generate_completion(sample: BfclExecSample, config: BfclExecRunConfig) -> BfclExecResult:
    prompt = build_prompt(sample)
    completion = chat_completion(
        base_url=config.base_url,
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        timeout_s=config.timeout_s,
    )
    passed, fail_reason, decoded, _details = evaluate_completion(sample, completion)
    return BfclExecResult(
        sample_index=sample.sample_index,
        task_id=sample.task_id,
        prompt=prompt,
        completion=completion,
        answer=json.dumps(decoded, ensure_ascii=False, sort_keys=True),
        reference_answer=json.dumps(sample.expected_executable_calls, ensure_ascii=False, sort_keys=True),
        is_passed=passed,
        fail_reason=fail_reason,
    )


def evaluate_samples(samples: Sequence[BfclExecSample], config: BfclExecRunConfig) -> list[BfclExecResult]:
    return [generate_completion(sample, config) for sample in samples]


def scoreboard_dataset_name(config: BfclExecRunConfig) -> str:
    dataset = config.scoreboard_dataset or f"{config.benchmark}_{config.split}"
    if config.limit is not None:
        dataset = f"{dataset}_limit{int(config.limit)}"
    dataset += dataset_sample_suffix(sample_size=config.sample_size, sample_seed=config.sample_seed)
    return dataset


def job_id(config: BfclExecRunConfig) -> str:
    return config.job_id or f"helicopter-{config.benchmark}"


def completion_sampling_config(config: BfclExecRunConfig) -> dict[str, Any]:
    return {
        "tool": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
        }
    }


def task_sampling_config(config: BfclExecRunConfig) -> dict[str, Any]:
    return {
        "avg_k": 1,
        "pass_ks": [1],
        "prompt_profile": "helicopter_bfcl_exec",
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "sampling_config": completion_sampling_config(config),
    }


def write_results(results: Sequence[BfclExecResult], *, config: BfclExecRunConfig, repo_root: Path) -> int:
    task_id = asyncio.run(
        write_scoreboard_results(
            [result.to_scoreboard() for result in results],
            config=ScoreboardWriteConfig(
                dataset=scoreboard_dataset_name(config),
                model=config.model,
                job_name=config.job_name,
                job_id=job_id(config),
                benchmark=config.benchmark,
                runner=config.runner,
                cot_mode=config.cot_mode,
                sampling_config=task_sampling_config(config),
                completion_sampling_config=completion_sampling_config(config),
            ),
            repo_root=repo_root,
        )
    )
    return int(task_id)


def run_bfcl_exec(config: BfclExecRunConfig, *, repo_root: Path) -> dict[str, Any]:
    samples = load_samples(config)
    results = evaluate_samples(samples, config)
    task_id = write_results(results, config=config, repo_root=repo_root)
    passed = sum(1 for result in results if result.is_passed)
    return {
        "task_id": task_id,
        "benchmark": config.benchmark,
        "dataset": scoreboard_dataset_name(config),
        "model": config.model,
        "category": config.category,
        "total": len(results),
        "passed": passed,
        "accuracy": passed / len(results) if results else 0.0,
    }


def dry_run_summary(config: BfclExecRunConfig) -> dict[str, Any]:
    return {
        "benchmark": config.benchmark,
        "source": "github://ShishirPatil/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data",
        "split": config.split,
        "category": config.category,
        "limit": config.limit,
        "sample_size": config.sample_size,
        "sample_seed": config.sample_seed if config.sample_size is not None else None,
        "base_url": config.base_url,
        "model": config.model,
        "scoreboard_dataset": scoreboard_dataset_name(config),
        "job_name": config.job_name,
        "job_id": job_id(config),
    }


def _force_array_prefix(sample: BfclExecSample) -> bool:
    return _is_parallel_record(sample) and len(sample.expected_executable_calls) > 1


def _complete_forced_prefix(sample: BfclExecSample, completion: str) -> str:
    if not _force_array_prefix(sample):
        return completion
    stripped = completion.lstrip()
    if stripped.startswith("["):
        return completion
    if stripped.startswith("{") and not completion.rstrip().endswith("]"):
        return "[\n" + completion.rstrip() + "\n]"
    return "[\n" + completion


def _is_parallel_record(sample: BfclExecSample) -> bool:
    return "parallel" in sample.category.lower()


def _execution_result_matches(
    actual: BfclExecCallResult,
    expected: BfclExecCallResult,
    match_type: str,
) -> tuple[bool, str]:
    if not expected.success:
        return False, f"expected_execution_error({expected.error})"
    if not actual.success:
        return False, f"decoded_execution_error({actual.error})"
    normalized = str(match_type or "exact_match").strip().lower()
    if normalized == "structural_match":
        return (True, "ok") if _same_structure(actual.result, expected.result) else (False, "structure_mismatch")
    if normalized == "real_time_match":
        return (
            (True, "ok")
            if _real_time_value_matches(actual.result, expected.result)
            else (False, "real_time_mismatch")
        )
    return (True, "ok") if _value_matches(actual.result, expected.result) else (False, "exact_mismatch")


def _same_structure(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return False
        return all(key in actual and _same_structure(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if not expected or not actual:
            return True
        return _same_structure(actual[0], expected[0])
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return True
    return type(actual) is type(expected) or isinstance(actual, type(expected))


def _real_time_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        actual_float = float(actual)
        expected_float = float(expected)
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        baseline = max(abs(expected_float), 1.0)
        return abs(actual_float - expected_float) / baseline <= 0.20
    return _value_matches(actual, expected) or _same_structure(actual, expected)


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if _both_plain_ints(actual, expected):
            return actual == expected
        actual_float = float(actual)
        expected_float = float(expected)
        if not math.isfinite(actual_float) or not math.isfinite(expected_float):
            return actual == expected
        return math.isclose(actual_float, expected_float, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return dict(actual) == dict(expected)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_value_matches(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _both_plain_ints(actual: Any, expected: Any) -> bool:
    return (
        isinstance(actual, int)
        and not isinstance(actual, bool)
        and isinstance(expected, int)
        and not isinstance(expected, bool)
    )


def _truthy_exec_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null"}
    return bool(value)


def _parse_call(text: str) -> tuple[str, tuple[Any, ...], dict[str, Any]]:
    parsed = ast.parse(str(text).strip(), mode="eval")
    if not isinstance(parsed.body, ast.Call):
        raise ValueError(f"not a function call: {text}")
    name = _call_name(parsed.body.func)
    args = tuple(_literal_from_ast(item) for item in parsed.body.args)
    kwargs = {
        keyword.arg: _literal_from_ast(keyword.value)
        for keyword in parsed.body.keywords
        if keyword.arg is not None
    }
    return name, args, kwargs


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_from_ast(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_from_ast(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_literal_from_ast(key): _literal_from_ast(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _literal_from_ast(node.operand)
        return -value if isinstance(value, (int, float)) else value
    if isinstance(node, ast.BinOp):
        left = _literal_from_ast(node.left)
        right = _literal_from_ast(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
    return ast.literal_eval(node)


def _python_literal(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_python_literal(item) for item in value) + "]"
    if isinstance(value, tuple):
        rendered = ", ".join(_python_literal(item) for item in value)
        return f"({rendered}{',' if len(value) == 1 else ''})"
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{_python_literal(key)}: {_python_literal(item)}" for key, item in value.items()) + "}"
    return repr(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _result_payload(result: BfclExecCallResult) -> dict[str, Any]:
    return {"call": result.call, "success": result.success, "result": result.result, "error": result.error or ""}


def _ground_truth_to_exec_calls(raw: Any) -> list[str]:
    calls: list[str] = []
    for item in _coerce_list(raw):
        if isinstance(item, str):
            text = item.strip()
            if text:
                calls.append(text)
            continue
        if not isinstance(item, Mapping) or len(item) != 1:
            continue
        name, raw_arguments = next(iter(item.items()))
        arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, Mapping):
            for key, value in raw_arguments.items():
                options = _coerce_list(value)
                arguments[str(key)] = options[0] if options else value
        calls.append(render_bfcl_exec_call({"name": str(name), "arguments": arguments}))
    return calls


__all__ = [
    "BfclExecRunConfig",
    "BfclExecSample",
    "BfclExecSandbox",
    "build_prompt",
    "dry_run_summary",
    "evaluate_completion",
    "evaluate_bfcl_exec_calls",
    "load_samples",
    "render_bfcl_exec_call",
    "run_bfcl_exec",
]
