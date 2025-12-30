# Table / RecordBatch / Schema / Field
이 문서는 nanosets의 `Table`/`RecordBatch`/`Schema`/`Field`가 어떤 역할을 하고, 왜 이런 구조가 Arrow-like(컬럼형, 버퍼 기반) 시스템에서 자연스러운지 설명합니다.

## 1. 큰 그림: 왜 `Table`을 한 덩어리가 아니라 여러 조각으로 나눌까요?
nanosets의 데이터 모델은 대략 이렇게 구성됩니다.

- `Field`: 컬럼 하나의 "정의" (이름/타입/nullable)
- `Schema`: 여러 `Field`를 모은 "테이블 구조"
- `RecordBatch`: "같은 스키마를 공유하는 컬럼 묶음" + "행 수(length)" (실제 데이터 보유)
- `Table`: 여러 `RecordBatch`를 이어붙인 "논리적 테이블"

이 구조의 핵심은 데이터(컬럼 버퍼)와 메타데이터(스키마)를 분리하고, 큰 테이블을 배치 단위로 쪼개서 관리한다는 점입니다.

### 왜 배치(`RecordBatch`)가 필요한가요?
대규모 데이터를 다루다 보면, "테이블 전체"를 한 번에 만들고 한 번에 처리하기 어렵습니다. 배치라는 단위를 두면 다음이 쉬워집니다.

- 부분 처리: 일부 구간만 slice/take/select
- concat: 여러 테이블을 batches만 이어붙여 합치기
- 메모리/캐시 친화성: 너무 큰 단일 블록을 피하고 적절한 덩어리 크기로 관리

결국 `Table`은 "논리적 전체", `RecordBatch`는 "물리적 단위"라는 느낌으로 보면 좋습니다.

## 2. `Schema`와 `Field`: 데이터의 정의
### `Field`는 무엇인가요?
`Field`는 컬럼 하나의 정의입니다.

- `name`: 컬럼 이름
- `dtype`: 컬럼 타입(`DataType`)
- `nullable`: null이 가능한지 여부

중요한 포인트: `Field`는 실제 데이터를 들고 있지 않습니다. 오직 "이 컬럼은 이런 이름/타입/nullable 규칙을 가진다"라는 메타데이터입니다.  
또한 `@dataclass(frozen=True)`라서 한 번 만들면 바뀌지 않습니다.

### `Schema`는 무엇인가요?
`Schema`는 `Field`들의 튜플을 모은 것입니다. 즉, "테이블은 이런 컬럼들로 구성된다"는 구조 정의입니다.

이때 `Schema`가 하는 일은 단순하지만 매우 중요합니다.

- 컬럼 순서 보장: `fields`의 순서가 곧 컬럼 순서
- 이름 → 인덱스 매핑: `schema.index("name")`
- 테이블/배치 간 일관성 체크의 기준

## 3. `RecordBatch`: "같은 길이의 컬럼들" 묶음
`RecordBatch`는 `schema`와 `columns`(Array 리스트)를 함께 들고, 같은 길이의 컬럼들이 모여 하나의 배치를 이룹니다.

### `RecordBatch`의 연산들

아래 연산들은 "row dict를 직접 조작"하기보다는, 각 컬럼 `Array`에 같은 연산을 적용해서 새로운 `RecordBatch`를 만드는 방식으로 동작합니다. 핵심은 `RecordBatch`가 "컬럼들의 묶음 + 같은 길이"라는 점입니다.

#### 1) `column(i_or_name)`
```python
def column(self, i_or_name: Union[int, str]) -> Array:
    if isinstance(i_or_name, int):
        return self.columns[i_or_name]
    if isinstance(i_or_name, str):
        idx = self.schema.index(i_or_name)
        return self.columns[idx]
    raise TypeError("Argument must be an integer index or a string column name.")
```

인덱스(`int`) 또는 컬럼명(`str`)로 특정 컬럼 `Array`를 가져옵니다. 이름이 들어오면 `Schema.index(name)`로 인덱스를 찾은 뒤 해당 컬럼을 반환합니다.

#### 2) `slice(offset, length)`

```python
def slice(self, offset: int, length: int) -> "RecordBatch":
    if length == 0:
        new_cols = [col.take([]) for col in self.columns]
        return RecordBatch(self.schema, new_cols)

    row_range = range(offset, offset + length)
    new_cols = [col.take(row_range) for col in self.columns]
    return RecordBatch(self.schema, new_cols)
```

연속 구간의 행을 뽑아 새 `RecordBatch` 를 만듭니다. 구현 관점에서 핵심은 다음 두 가지입니다.

- `range(offset, offset + length)`라는 "행 인덱스 범위"를 만들고
- 모든 컬럼에 대해 `col.take(row_range)`를 호출해, 같은 행 구간을 추출합니다.

길이가 0이면 빈 결과를 만들기 위해 모든 컬럼에 `take([])`를 적용합니다.

#### 3) `take(indices)`

```python
def take(self, indices: Sequence[int]) -> "RecordBatch":
    new_cols = [col.take(indices) for col in self.columns]
    return RecordBatch(self.schema, new_cols)
```

임의의 행 인덱스 목록(또는 시퀀스)을 받아 그 행들만 골라 새 `RecordBatch` 를 만듭니다.  
구현은 `slice`와 동일한 패턴으로, 컬럼마다 `col.take(indices)`를 적용합니다.

#### 4) `select(names)`
```python
def select(self, names: List[str]) -> "RecordBatch":
    field_indices = [self.schema.index(name) for name in names]

    new_fields = tuple(self.schema.fields[i] for i in field_indices)
    new_schema = Schema(new_fields)

    new_columns = [self.columns[i] for i in field_indices]
    return RecordBatch(new_schema, new_columns)
```

컬럼 일부만 선택해 새 `RecordBatch` 를 만듭니다. 핵심은 "스키마와 컬럼을 동시에 같은 방식으로 필터링"하는 것입니다.

- 선택된 이름들로부터 field index들을 만들고
- 그 인덱스들로 `Schema(fields)`를 재구성한 뒤
- 동일 인덱스로 `columns`도 뽑아 새 배치를 만듭니다.

#### 5) `to_list()`

```python
def to_list(self) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    per_column_lists = [col.to_list() for col in self.columns]

    for row_index in range(self.length):
        row: Dict[str, Any] = {}
        for field, column_values in zip(self.schema.fields, per_column_lists):
            row[field.name] = column_values[row_index]
        rows.append(row)

    return rows
```

`RecordBatch`를 "rows-only" 형태인 `List[Dict[str, Any]]`로 변환합니다. 핵심은 다음 흐름입니다.

1) 각 컬럼을 `col.to_list()`로 파이썬 리스트로 바꿔서 컬럼별 리스트를 준비한 뒤  
2) `row_index`를 돌면서 각 필드 이름에 해당 컬럼 값을 채워 row dict를 구성합니다.

#### 6) `from_list(rows)`

```python
@classmethod
def from_list(cls, rows: List[Optional[Dict[str, Any]]], strict_keys: bool = False) -> "RecordBatch":
    struct = StructArray.from_list(rows, strict_keys=strict_keys)

    fields = tuple(
        Field(
            name=name,
            dtype=child.dtype,
            nullable=(child.validity is not None),
        )
        for name, child in zip(struct.field_names, struct.children)
    )
    schema = Schema(fields)
    return cls(schema, struct.children)
```

row 기반 입력(`List[Optional[Dict[str, Any]]]`)을 받아 컬럼형으로 재구성해 `RecordBatch`를 만듭니다. 핵심은 "row들을 먼저 `StructArray`로 만들고, 그 children이 곧 컬럼들"이라는 점입니다.

- `StructArray.from_list(rows, ...)`로 row dict들을 구조체 컬럼(여러 children 컬럼)로 변환
- 각 child 컬럼의 `dtype`와 `validity`를 보고 `Field`를 구성
- `Schema(fields)` + `struct.children`로 `RecordBatch(schema, columns)` 생성

요약하면, `RecordBatch`의 대부분 연산은 "행 단위로 뭔가를 하는 것처럼 보이지만", 실제로는 컬럼 `Array`에 동일한 인덱싱 연산을 적용해서 새 배치를 만들어내는 패턴이라고 이해하시면 됩니다.

## 4. `Table`: 여러 `RecordBatch`를 이어붙인 논리적 테이블
`Table`은 `schema`가 모두 동일한 `batches: List[RecordBatch]`를 모아 논리적 테이블을 이룹니다.

### `Table`이 제공하는 관점
- 논리적 길이: `self.length = sum(b.length for b in batches)`
- 배치 반복: `iter_batches()`
- 컬럼 접근: `column(i_or_name)`가 "배치별 컬럼 리스트"를 반환
  - 즉, Table의 컬럼은 단일 Array가 아니라 (배치 수만큼의 Array 리스트) 로 존재합니다.

### 간단한 연산들

`Table`의 연산은 기본적으로 "여러 `RecordBatch` 위에서" 동작합니다. 즉, `RecordBatch`가 물리적 단위라면 `Table`은 그 위에 전역 인덱스(테이블 전체 기준) 를 제공하는 래퍼라고 보시면 됩니다.

#### 1) `__getitem__`
```python
def __getitem__(self, item):
    if isinstance(item, int):
        return self.slice(item, 1).to_list()[0]
    elif isinstance(item, slice):
        indices = list(range(*item.indices(len(self))))
        return self.take(indices)
    else:
        raise TypeError("Invalid argument type.")
```

- `table[i]`는 전역 인덱스 `i`를 기준으로 1행짜리 slice를 만든 뒤 `to_list()[0]`로 row dict를 반환합니다.
- `table[a:b:c]`는 파이썬 slice를 전개해서 인덱스 리스트를 만들고, 내부적으로 `take(indices)`를 호출합니다.

#### 2) `column(i_or_name)`

```python
def column(self, i_or_name) -> List[Array]:
    cols: List[Array] = []
    for b in self.batches:
        cols.append(b.column(i_or_name))
    return cols
```

`Table`의 컬럼은 단일 `Array`가 아니라, 배치마다 하나씩 존재합니다. 그래서 `column(...)`은 `List[Array]`를 반환합니다.


#### 3) `select(names)`
`select`는 모든 배치에 대해 `RecordBatch.select(names)`를 적용한 결과를 다시 `Table`로 묶습니다.

```python
def select(self, names: List[str]) -> "Table":
    new_batches = [b.select(names) for b in self.batches]
    return Table.from_batches(new_batches)
```

#### 4) `concat(tables)`
`concat`은 "row를 다시 만들거나 재배치"하지 않고, 배치 리스트만 이어붙여 새로운 `Table`을 만듭니다. (단, 스키마는 모두 같아야 합니다.)

```python
@classmethod
def concat(cls, tables: List["Table"]) -> "Table":
    batches: List[RecordBatch] = []
    for table in tables:
        batches.extend(table.batches)
    return cls.from_batches(batches)
```

## 5. `table.slice(offset, length)`

`Table.slice`의 목표는 "테이블 전체를 하나의 연속된 행 배열처럼" 보이게 하면서도, 
내부적으로는 여러 `RecordBatch`에 쪼개져 있는 구조를 유지한 채로 필요한 배치 구간만 잘라서 새 `Table`을 만드는 것입니다.

핵심 아이디어는 간단합니다.

- 사용자가 준 `offset`, `length`는 전역(whole table) 기준
- 각 `RecordBatch`는 자기 배치 안에서의 로컬(local) 인덱스만 이해합니다
- 그래서 global → local 인덱스 변환을 계산해서, 각 배치에서 필요한 부분만 `slice`로 잘라냅니다.


```python
remaining = length
batch_start_global = 0
new_batches: List[RecordBatch] = []

for batch in self.batches:
    batch_length = batch.length
    batch_end_global = batch_start_global + batch_length

    if batch_end_global <= offset:
        batch_start_global = batch_end_global
        continue

    local_start = max(0, offset - batch_start_global)
    local_available = batch_length - local_start
    local_len = min(remaining, local_available)

    new_batches.append(batch.slice(local_start, local_len))
    remaining -= local_len

    if remaining <= 0:
        break

    batch_start_global = batch_end_global

return Table.from_batches(new_batches)
```

### 1) `batch_start_global`와 `batch_end_global`: 전역 인덱스
배치가 여러 개 있을 때, 각 배치는 전역적으로 이런 구간을 차지합니다.

- 배치 0: `[0, len0)`
- 배치 1: `[len0, len0+len1)`
- 배치 2: `[len0+len1, len0+len1+len2)`
- ...

코드는 이걸 매번 계산하기 위해 `batch_start_global`(시작)과 `batch_end_global`(끝)을 추적합니다.

- `batch_start_global`: 현재 배치가 테이블에서 시작하는 전역 행 인덱스
- `batch_end_global = batch_start_global + batch_length`: 현재 배치의 전역 끝(Exclusive)

### 2) `remaining`: 아직 더 잘라야 하는 행 수

예를 들어 `slice(offset=5, length=10)`이면 총 10행을 잘라내야 합니다.
- 배치0에서 3행을 잘라냈으면 `remaining`은 7이 되고
- 배치1에서 5행을 잘라냈으면 `remaining`은 2가 됩니다.
- 처음에는 하나도 안잘렸으니 `remaining=10=length`입니다.
- 만약 어떤 배치에서 일부를 잘라냈으면 그만큼 `remaining`을 줄입니다.

### 3) `if batch_end_global <= offset`: 이 배치는 slice 시작보다 앞에 있음
이 조건은 "현재 배치가 slice 시작점(offset)보다 완전히 앞에 있는가?"를 체크합니다.

- 현재 배치 전역 구간이 `[batch_start_global, batch_end_global)`
- slice는 `offset`부터 시작
- 만약 `batch_end_global <= offset`이면
  현재 배치의 끝이 offset보다 같거나 앞이므로, 이 배치는 slice에 전혀 포함되지 않습니다.

그래서 그냥 건너뛰고 다음 배치로 넘어갑니다.

```python
if batch_end_global <= offset:
    batch_start_global = batch_end_global
    continue
```

여기서 `batch_start_global`을 업데이트하는 이유는 다음 배치 전역 범위를 정확히 계산하기 위해서입니다.

### 4) `local_start`: 이 배치 내부에서 어디서부터 자를지
이제 "slice가 이 배치에 포함되기 시작"하는 상황입니다.

- slice 전역 시작은 `offset`
- 배치 전역 시작은 `batch_start_global`

배치 내부 로컬 시작은:
- `offset - batch_start_global` (전역 offset을 배치 기준으로 옮긴 값)
- 예를 들어 `offset=10`, `batch_start_global=7`이면
  - `current_batch[3:]`가 slice에 포함됩니다.
  - 그래서 `local_start = 10 - 7 = 3`이 됩니다.
  - 즉, 이 배치에서 3번째 행부터 slice가 시작됩니다.
- 단, offset이 배치 시작보다 앞에 있을 수도 있습니다(예: slice가 이전 배치에서 시작해서 이어지는 경우).
  그때는 현재 배치에서 로컬 시작은 0 보다는 작을 수 없으므로 `max(0, ...)`를 씁니다.

```python
local_start = max(0, offset - batch_start_global)
```

이 한 줄이 "전역 offset을 배치의 로컬 인덱스로 투영(projection)하는 핵심"입니다.

### 5) `local_len`: 이 배치에서 실제로 몇 행을 가져올지
이 배치에서 가져올 수 있는 최대 길이는:

- 배치 안에서 `local_start`부터 끝까지 남은 행 수
- 즉 `local_available = batch_length - local_start`

하지만 전체 slice에서 아직 필요한 길이는 `remaining` 입니다.
그래서 둘 중 작은 값만큼만 이 배치에서 가져옵니다.

```python
local_available = batch_length - local_start
local_len = min(remaining, local_available)
```

이렇게 하면 배치 경계를 넘어가지 않고 slice가 필요한 만큼만 정확히 가져오게 됩니다.

### 6) 배치 slice를 수행하고, 남은 길이를 갱신
```python
new_batches.append(batch.slice(local_start, local_len))
remaining -= local_len
```

여기서 중요한 점은 `Table.slice`는 "테이블의 컬럼들을 직접 만지지 않고" 각 배치에 대해 `RecordBatch.slice`를 호출해서 새 배치들을 만들고, 그걸 다시 `Table`로 묶는다는 것입니다.

즉, `Table`은 배치 단위로 잘라붙이는 역할을 하고,
실제 컬럼 단위의 잘라내기는 `RecordBatch.slice`가 합니다.

### 7) 다 채웠으면 종료
```python
if remaining <= 0:
    break
```

필요한 길이를 다 채웠으면 이후 배치를 볼 필요가 없습니다.

### 8) 예시

배치 길이가 `[3, 5, 2]`라고 해봅시다.

- 배치0 전역 `[0,3)`
- 배치1 전역 `[3,8)`
- 배치2 전역 `[8,10)`

`slice(offset=2, length=6)`이면 전역 구간은 `[2,8)`입니다.

- 배치0: `[0,3)`과 겹치는 부분은 `[2,3)` → 로컬 시작 2, 길이 1
- 배치1: `[3,8)`과 겹치는 부분은 `[3,8)` → 로컬 시작 0, 길이 5
- 배치2: `[8,10)`은 slice 끝 8과 같으니 포함 없음

결과적으로:
- 배치0에서 1행
- 배치1에서 5행
을 잘라서 새 Table을 만듭니다.

### 9) 이 방식의 장점
- 사용자 입장: `Table`을 하나의 큰 배열처럼 `slice(offset, length)` 사용 가능
- 구현/성능 관점: 내부 배치 구조를 유지하면서 필요한 배치 조각만 새로 만들어 구성 가능
- Arrow-like 컬럼형 관점: "행 단위로 하나씩 복사"가 아니라 "배치/컬럼 단위 연산을 조합"하는 형태로 설계가 자연스럽습니다

## 6. `table.take(indices)`

이 함수의 목적은 `Table` 전체를 "전역 인덱스(0..len(table)-1)" 관점에서 `take` 하는 것입니다. 그런데 `Table`은 내부적으로 여러 `RecordBatch`로 나뉘어 있으니, 전역 인덱스들을 배치별 로컬 인덱스로 바꿔서 각 배치에 `RecordBatch.take(local_indices)`를 적용한 뒤, 그 결과 배치들을 다시 `Table`로 묶습니다.

### 1) 빈 입력 처리
`indices`가 비어 있으면 결과는 "행 0개짜리 테이블"이어야 합니다. 구현에서는 첫 배치의 컬럼들에 대해 `col.take([])`를 해서 스키마는 유지하되 행이 없는 컬럼들을 만들고, 그걸로 `RecordBatch`를 만든 뒤 반환합니다.

```python
if not indices:
    first_batch = self.batches[0]
    empty_columns = [col.take([]) for col in first_batch.columns]
    empty_batch = RecordBatch(self.schema, empty_columns)
    return Table.from_batches([empty_batch])
```

핵심 포인트는 "스키마 유지"입니다. 행은 0개여도 컬럼 정의는 남겨둬야 이후 연산/IO에서 일관성이 좋아집니다.

### 2) 전역 인덱스 정규화: 음수 인덱스 허용
파이썬처럼 `-1` 같은 인덱스를 허용하려고, 먼저 모든 인덱스를 `[0, n)` 범위로 정규화합니다.

```python
n = self.length
normalized_indices = [normalize_index(idx, n) for idx in indices]
```

여기서 `normalize_index`는 보통:
- 음수면 `idx += n`
- 범위 밖이면 에러
같은 역할을 한다고 보면 됩니다.

### 3) `batch_starts`: 전역 인덱스를 배치로 매핑하기 위한 기준
각 배치가 테이블에서 시작하는 전역 위치를 모읍니다.

예를 들어 배치 길이가 `[3, 5, 2]`면:
- `batch_starts = [0, 3, 8]`
이 됩니다.

```python
batch_starts: List[int] = []
current = 0
for batch in self.batches:
    batch_starts.append(current)
    current += batch.length
```

이걸 만들어두면, 전역 인덱스 `idx`가 들어왔을 때 "어느 배치에 속하는지"를 빠르게 찾을 수 있습니다.

### 4) `bisect_right`: "idx가 속한 배치" 찾기

```python
batch_idx = bisect_right(batch_starts, idx) - 1
```

`bisect_right(batch_starts, idx)`는 `idx`를 삽입했을 때 정렬이 유지되는 "삽입 위치"를 주는데, 
그 위치에서 1을 빼면 "idx 이하인 시작점 중 가장 오른쪽"을 가리키게 됩니다. 즉, 그게 idx가 속한 배치가 됩니다.

예시: `batch_starts = [0, 3, 8]`
- `idx = 0`  → `bisect_right(...)=1` → `batch_idx=0`
- `idx = 2`  → `bisect_right(...)=1` → `batch_idx=0`
- `idx = 3`  → `bisect_right(...)=2` → `batch_idx=1`
- `idx = 7`  → `bisect_right(...)=2` → `batch_idx=1`
- `idx = 8`  → `bisect_right(...)=3` → `batch_idx=2`

그 다음 로컬 인덱스로 바꿉니다.

```python
local_idx = idx - batch_starts[batch_idx]
```

### 5) 왜 "그룹핑"을 하나요? (`current_batch_idx`, `flush`)
전역 인덱스를 하나씩 처리하면서, 같은 배치에 속한 것들을 모아두었다가 한 번에 `RecordBatch.take()`를 호출합니다.

이렇게 하면 같은 배치에 대한 `take()` 호출을 여러 번 하는 대신 한 번에 모아서 처리할 수 있습니다.

이를 위해 상태를 들고 갑니다.

- `current_batch_idx`: 지금 모으고 있는 배치 번호
- `current_local_indices`: 그 배치에서 뽑을 로컬 인덱스들
- `prev_local`: 바로 직전 로컬 인덱스(연속성 체크용)

그리고 `flush()`가 "지금까지 모은 것들을 실제 배치로 변환해서 new_batches에 추가"하는 역할을 합니다.

```python
def flush() -> None:
    nonlocal current_batch_idx, current_local_indices, prev_local
    if current_batch_idx is None or not current_local_indices:
        return
    base_batch = self.batches[current_batch_idx]
    new_batches.append(base_batch.take(current_local_indices))
    current_batch_idx = None
    current_local_indices = []
    prev_local = None
```

### 6) 연속 인덱스 최적화: 같은 배치 + (prev_local + 1)이면 계속 누적
루프에서 핵심 분기:

```python
if batch_idx == current_batch_idx and prev_local is not None and local_idx == prev_local + 1:
    current_local_indices.append(local_idx)
    prev_local = local_idx
else:
    flush()
    current_batch_idx = batch_idx
    current_local_indices = [local_idx]
    prev_local = local_idx
```

이 로직은 "같은 배치에서 로컬 인덱스가 연속"이면 계속 같은 묶음으로 유지합니다. 연속이 깨지면(다른 배치로 가거나, 같은 배치지만 건너뛰는 인덱스가 나오면) 이전 묶음을 `flush()`해서 확정하고 새 묶음을 시작합니다.

주의: 이 구현은 입력 indices의 순서를 그대로 따라갑니다.
- indices가 정렬되어 있으면 이런 그룹핑이 잘 먹습니다.
- indices가 뒤죽박죽이면 flush가 자주 일어나고 결과도 그 순서대로 묶입니다(그래도 의미는 맞긴 합니다).

### 7) 루프 끝난 뒤 마지막 flush
루프 안에서 flush는 "새 그룹 시작할 때"만 일어나므로, 끝나고 나서 한 번 더 flush해야 마지막 그룹이 반영됩니다.

```python
flush()
```

### 8) 결과가 비면 다시 빈 테이블 처리
모종의 이유로 `new_batches`가 비어 있으면(예: indices가 결국 비거나 처리되지 않은 경우를 방어) 앞에서와 같은 방식으로 빈 테이블을 만들어 반환합니다.

```python
if not new_batches:
    first_batch = self.batches[0]
    empty_columns = [col.take([]) for col in first_batch.columns]
    empty_batch = RecordBatch(self.schema, empty_columns)
    return Table.from_batches([empty_batch])
return Table.from_batches(new_batches)
```

### 9) 요약
- 전역 인덱스 `indices`를 배치별 로컬 인덱스로 바꾼다
- 같은 배치(특히 연속 인덱스)는 묶어서 `RecordBatch.take()`를 호출한다
- 그렇게 만든 `RecordBatch`들을 이어붙여 새 `Table`을 만든다

## 7. Table / RecordBatch / Schema / Field: 3줄 요약
- `Field`/`Schema`는 컬럼 정의(이름/타입/nullable)이고, 배치/테이블의 일관성을 보장하는 기준입니다.
- `RecordBatch`는 동일 길이의 컬럼(Array)들을 스키마와 함께 묶은 단위이며, slice/take/select 같은 연산의 기본 단위가 됩니다.
- `Table`은 여러 `RecordBatch`를 이어붙인 논리적 테이블이며, 배치 단위 구조는 대용량 처리와 IPC(mmap, Zero-copy 지향) 같은 시스템과 자연스럽게 맞물립니다.