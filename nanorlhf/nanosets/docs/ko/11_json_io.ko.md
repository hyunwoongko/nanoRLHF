# JSON / JSONL IO (Rows-only)
이 문서에서는 `Table`/`RecordBatch`를 row 기반 JSON 포맷(rows-only)으로 내보내고 다시 읽어오는 유틸 함수들을 설명합니다. nanosets의 내부 표현은 Arrow-like 컬럼형 구조(`values`/`offsets`/`validity`/`indices` 등 버퍼 기반)인데, JSON/JSONL은 그런 버퍼 레이아웃을 보존하기보다 사람이 읽기 쉬운 row(dict) 표현으로 변환하는 데 목적이 있습니다.

## 1. rows-only JSON이 뭔가요?
여기서 "rows-only"는 JSON의 루트가 row들의 리스트라는 뜻입니다.

- 각 row는 `dict` (예: `{"name": "Kevin", "age": 30}`) 이거나
- null row라면 `None` (JSON에서는 `null`) 입니다.

컬럼형 스키마/버퍼 같은 정보는 JSON에 같이 저장되지 않습니다. 대신 "row를 파이썬 객체로 materialize해서 JSON으로 덤프한다"에 초점이 있습니다.
즉 JSON으로 나가면 컬럼형 이점(연속 버퍼, 타입 안정성, zero-copy) 을 대부분 내려놓고, 대신 가독성과 범용성을 얻습니다.

## 2. 타입 정의: `Row`, `TableLike`
이 모듈은 두 가지 입력 타입을 다룹니다.

- `TableLike = Union[Table, RecordBatch]`
- `Row = Optional[Dict[str, Any]]`

즉, 함수들은 `Table` 전체나 `RecordBatch` 하나를 받아서 row 단위로 읽고/쓰는 방식으로 통일합니다.

## 3. `iter_rows(obj)`: row 스트리밍 인터페이스
`iter_rows(obj)`는 `Table`/`RecordBatch`에서 row를 하나씩 `yield` 합니다.

- `obj`가 `RecordBatch`면 `obj.to_list()`의 row들을 그대로 순회
- `obj`가 `Table`이면 `obj.batches`를 순서대로 돌면서 각 `batch.to_list()`를 이어붙여 순회

## 4. `materialize(obj)`: 모든 row를 리스트로 만들기
`materialize(obj)`는 내부적으로 `list(iter_rows(obj))`를 호출해 모든 row를 한 번에 메모리에 올립니다.

### 왜 이런 함수가 필요한가요?
어떤 출력 포맷/라이브러리는 "전체 리스트" 형태를 요구합니다. 대표적으로 `to_json`이 그렇습니다.

- `to_json`은 JSON 루트가 "리스트"인 형태를 만들기 때문에, 구현이 가장 단순한 방식은 먼저 rows 리스트를 만들어두고 `json.dump(rows, ...)`를 하는 것입니다.

반대로 `to_jsonl`은 한 줄씩 쓰면 되므로 굳이 전부 materialize할 필요가 없습니다.

## 5. `to_json(fp, obj, indent=2)`: JSON 배열로 쓰기
`to_json`은 rows-only JSON을 씁니다.

- 먼저 `rows = materialize(obj)`로 row 리스트를 만들고
- `json.dump(rows, fp, ensure_ascii=False, indent=indent)`를 호출합니다.

## 6. `to_jsonl(fp, obj)`: JSONL로 쓰기
`to_jsonl`은 row를 한 줄씩 씁니다.

- `for row in iter_rows(obj):`
  - `fp.write(json.dumps(row, ensure_ascii=False))`
  - `fp.write("\n")`

## 7. `from_json(path, batch_size=DEFAULT_BATCH_SIZE)`: JSON 배열 읽기
`from_json`은 파일을 열고 `json.load(f)`로 전체를 읽은 뒤, `Table.from_list(data, batch_size=batch_size)`를 호출합니다.

여기서 포인트는:

- 입력 JSON은 이미 row 리스트라서, 결국 `Table`을 다시 만들 때도 row 기반 생성(`from_list`)을 사용합니다.
- `batch_size`는 `Table` 내부가 `RecordBatch`들로 구성되기 때문에 "몇 row씩 끊어서 배치를 만들지"를 결정합니다.

### `batch_size`는 왜 있나요?
너무 큰 한 덩어리로 만들면 메모리/캐시 측면에서 부담이 커질 수 있고, 너무 작게 쪼개면 overhead가 생길 수 있습니다. 그래서 "적당히 몇 줄씩 끊는가"가 옵션으로 등장합니다.

## 8. `from_jsonl(path, batch_size=DEFAULT_BATCH_SIZE)`: JSONL 읽기
`from_jsonl`은 각 줄을 `json.loads(line)`로 파싱해서 `rows: List[Row]`에 모은 뒤 `Table.from_list(rows, batch_size=batch_size)`를 호출합니다.

## 9. 언제 이 IO를 쓰고, 언제 IPC를 쓰나요?
### 이 JSON/JSONL IO가 좋은 경우
- 디버깅: `Table` 내용을 사람이 바로 확인하고 싶을 때
- 범용 교환: 다른 언어/환경과 "일단 row dict"로 주고받고 싶을 때
- 단순 저장: 성능/용량보다 편의성이 중요할 때

### IPC가 좋은 경우
- 컬럼형 버퍼 레이아웃을 보존하고 싶을 때
- `mmap` 기반으로 read 경로를 zero-copy에 가깝게 만들고 싶을 때
- 대용량에서 성능/메모리 효율이 중요할 때

## 10. JSON / JSONL IO: 3줄 요약
- 이 모듈은 `Table`/`RecordBatch`를 row(dict)/null 형태로 변환해 JSON 또는 JSONL로 저장/로드합니다.
- `to_json`은 JSON 배열을 만들기 위해 `materialize`로 rows 리스트를 먼저 만들고, `to_jsonl`은 `iter_rows`로 한 줄씩 스트리밍합니다.
- Arrow-like 컬럼형 버퍼 구조를 보존하려는 목적이면 IPC가 맞고, JSON/JSONL은 디버깅/교환용으로 유용합니다.
