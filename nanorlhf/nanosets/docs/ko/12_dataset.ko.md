# Dataset

이 문서에서는 `Dataset`이 무엇인지와, `Table`을 기반으로 어떤 기능을 제공하는지 설명합니다. `Dataset`은 내부에 `Table` 하나를 보관하고, 그 위에 자주 쓰는 작업(선택, 섞기, 변환, 필터링, 저장/로드)을 얹은 형태입니다.

## 1. 구조와 역할

`Dataset`은 다음처럼 `Table`을 감싸는 구조입니다.

```python
class Dataset:
    def __init__(self, table: Table):
        self.table = table
```

따라서 `Dataset`의 동작은 크게 두 부류로 나눠서 이해하면 편합니다.

### 1) Table의 연산을 그대로 활용하는 부류
row를 따로 만들지 않고, 배치/컬럼 구조를 유지한 채로 동작합니다.

- `__len__`
- `select`
- `shuffle`
- `select_columns`
- `remove_columns`
- `save_to_disk`의 저장 자체(버퍼 기반 직렬화)

### 2) row(dict)로 변환한 다음 다시 구성하는 부류
파이썬 함수 적용이나 JSON 출력 같은 이유로, 중간에 row 기반 표현으로 변환합니다.

- `__getitem__`
- `to_dict`
- `to_json`
- `map`
- `filter`
- JSON 계열 로드(`from_json/from_jsonl`)는 입력 자체가 row 기반이어서 결국 비슷한 흐름을 갖습니다

## 2. 길이와 출력

### 길이
Dataset의 길이는 Table의 전체 행 수입니다.

```python
def __len__(self) -> int:
    return self.table.length
```

### 출력 표현
디버깅 시에 행 수와 스키마를 간단히 보여줍니다.

```python
def __repr__(self) -> str:
    return f"Dataset(num_rows={len(self)}, schema={self.table.schema})"
```

## 3. 인덱싱과 슬라이싱

### 정수 인덱싱: dataset[i]
정수 인덱싱은 1행을 선택한 뒤 row dict로 반환합니다.

```python
return self.select([item]).to_dict()[0]
```

- 음수 인덱스는 파이썬 규칙에 맞게 보정합니다
- 1개 행을 `select([i])`로 선택합니다
- `to_dict()`에서 row 기반 리스트로 변환합니다
- 그 중 첫 행을 반환합니다

### 슬라이싱: dataset[a:b:c]
슬라이스는 인덱스 목록을 만들고 선택한 뒤, row dict 리스트를 반환합니다.

```python
indices = list(range(*item.indices(len(self))))
return self.select(indices).to_dict()
```

## 4. 선택 연산: select와 shuffle

### select
`Dataset.select(indices)`는 내부적으로 `Table.take(indices)`를 호출합니다.

```python
def select(self, indices: Sequence[int]) -> "Dataset":
    return Dataset(self.table.take(indices))
```

`Table.take`는 전역 인덱스를 배치 단위로 나눠서 각 `RecordBatch.take`를 호출하고, 그 결과 배치들을 다시 `Table`로 묶는 방식입니다. Dataset은 그 결과를 감싸서 반환합니다.

### shuffle
`shuffle`은 0..n-1 인덱스를 만든 뒤 랜덤으로 섞고 `select`로 재배열합니다.

```python
idx = list(range(len(self)))
rng.shuffle(idx)
return self.select(idx)
```

## 5. 컬럼 선택과 제거

### select_columns
지정한 컬럼만 남긴 Dataset을 만듭니다.

```python
def select_columns(self, column_names: List[str]) -> "Dataset":
    return Dataset(self.table.select(column_names))
```

### remove_columns
제거할 컬럼을 제외한 나머지 컬럼을 선택합니다.

```python
all_names = self.table.column_names()
keep = [name for name in all_names if name not in drop_set]
return Dataset(self.table.select(keep))
```

이 두 연산은 row로 변환하지 않고, 스키마와 컬럼 참조 구성을 바꾸는 형태로 진행됩니다.

## 6. to_dict와 to_json

### to_dict
전체 데이터를 row dict 리스트로 변환합니다.

```python
def to_dict(self) -> List[Optional[dict]]:
    return self.table.to_list()
```

### to_json
`lines=True`면 JSONL(한 줄에 한 row), `lines=False`면 JSON(하나의 큰 리스트)로 저장합니다.

```python
if lines:
    to_jsonl(fp, self.table)
else:
    to_json(fp, self.table)
```

## 7. 저장과 로드

### save_to_disk: nano(IPC) 저장
Table을 IPC 포맷으로 저장합니다.

```python
with open(path, "wb") as fp:
    write_table(fp, self.table)
```

이 경로는 `.nano` 파일을 만들어 `read_table`에서 `mmap` 기반으로 로드할 수 있게 하는 목적을 가집니다.

### load_dataset: 확장자에 따라 로드
입력 파일의 확장자에 따라 로더를 선택합니다.

```python
if e == "json": return from_json(...)
if e in ("jsonl", "ndjson"): return from_jsonl(...)
if e == "nano": return read_table(...)
```

여러 파일이 들어오면 각 파일을 Table로 읽고, 필요하면 `Table.concat`으로 이어붙입니다.

또한 `load_from_disk = load_dataset`는 별칭입니다.

## 8. map

`map`은 두 모드를 지원합니다.

### batched=False
배치별로 row 리스트로 변환한 뒤, row 하나씩 함수에 넣고 결과 row들을 다시 `RecordBatch.from_list`로 구성합니다.

```python
rows = batch.to_list()
out_rows = [function(row) for row in rows]
new_batches.append(RecordBatch.from_list(out_rows))
```

### batched=True
row들을 버퍼에 모아두었다가, 버퍼가 어느 정도 차면 리스트 단위로 함수를 호출합니다. 이때 함수는 리스트를 받아 리스트를 반환해야 합니다.

```python
mapped = function(buffer)
if not isinstance(mapped, list):
    raise TypeError(...)
new_batches.append(RecordBatch.from_list(mapped))
```

마지막에 새 배치들을 `Table.from_batches`로 묶어 Dataset을 반환합니다.

## 9. filter

`filter`는 row dict를 predicate로 검사해서 통과한 row만 모읍니다. 일정 크기(`batch_size`)가 되면 `RecordBatch.from_list`로 배치를 만들고, 끝까지 처리한 뒤 남은 버퍼도 배치로 만듭니다.

결과가 비어 있으면(모두 걸러진 경우) 스키마를 유지한 "0행 테이블"을 만들기 위해 빈 컬럼들을 구성해 반환합니다.

## 10. map과 filter의 공통점
- 중간에 row(dict) 표현으로 변환합니다.
- 이 과정중에 materialize 되어 메모리를 추가로 사용합니다.
- 그런데 유저의 '임의의 함수'를 받는 특성상 이것은 불가피합니다.
- 사실 이런 문제를 피하려면 arrow의 `compute` 같은 저수준 연산을 써야 하지만, 유저 입장에서는 사용성이 크게 떨어집니다.
- 실제로 Hugging Face `datasets`에서도 사용성을 더 중요시 하기에 `map`과 `filter`는 내부적으로 row 기반 변환을 거칩니다.

## 11. 요약

- `Dataset`은 `Table`을 감싸는 형태로, 선택/셔플/컬럼 선택 같은 작업은 주로 `Table.take/select`를 이용합니다.
- `.nano` 저장과 로드는 IPC 포맷을 통해 `mmap` 기반 로드 경로를 사용할 수 있게 합니다.
- `map/filter/to_dict/to_json/__getitem__` 같은 기능은 row(dict) 기반 변환을 포함하며, 변환 결과를 다시 `RecordBatch/Table`로 구성합니다.