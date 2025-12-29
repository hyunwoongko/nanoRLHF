# Validity Bitmap
이 문서에서는 Validity Bitmap에 대해 설명합니다.

## 1. Validity Bitmap은 무엇인가요?
Validity Bitmap은 결측치(null 값)를 효율적으로 관리하기 위한 방법입니다.
실세계 데이터셋에는 많은 결측치 (None, NaN 등)가 포함될 수 있습니다.
만약 이러한 결측치를 데이터와 함께 저장한다면 서로 다른 Python 데이터 타입이 섞이게 됩니다.

다음과 같은 데이터를 생각해봅시다.

```python
data = [10, None, 30, 40]
```

Python은 내부적으로 이들을 오브젝트 포인터 (PyObject*)로 저장하게 되며 
이들은 연속되지 않은 메모리 공간에 흩어질 가능성이 생깁니다.

| Index | Value | Actual Storage Type            |
|-------|-------|--------------------------------|
| 0     | 10    | PyObject* (points to int)      |
| 1     | None  | PyObject* (points to NoneType) |
| 2     | 30    | PyObject* (points to int)      |
| 3     | 40    | PyObject* (points to int)      |

이렇게 조각난 메모리 레이아웃은 다음과 같은 몇가지 이슈를 발생시킵니다.

- SIMD 최적화가 어렵습니다.
- 캐시 효율성이 떨어집니다.
- PyObject 오버헤드가 발생합니다.

이를 해결하기 위해 결측치를 별도로 보관하는 방식에 대해 생각 해볼 수 있습니다.

```python
values = [10, 0, 30, 40]
validity = [1, 0, 1, 1]  # 1은 유효한 값, 0은 결측치를 나타냅니다.
```

이렇게 값과 결측치를 분리함으로써, 숫자 데이터는 컴팩트하고 연속된 메모리 블록(int32, float64 등)으로 저장될 수 있습니다.
우리는 이것을 바이트 단위 (`bytearray`) 로 저장하여 아예 물리적으로 연속적인 공간에 데이터를 저장할 수 있습니다.
그러므로서 위에서 언급된 세가지 이슈를 해결할 수 있습니다.

```python
values = bytearray([10, 0, 30, 40])  # int로 연속된 메모리 레이아웃
validity = bytearray([0b00001011])  # Bitmap으로 결측치 정보 저장
```

| Index | Value | Actual Storage Type               |
|-------|-------|-----------------------------------|
| 0     | 10    | int (stored in contiguous memory) |
| 1     | 0     | int (stored in contiguous memory) |
| 2     | 30    | int (stored in contiguous memory) |
| 3     | 40    | int (stored in contiguous memory) |

## 2. 왜 연속된 메모리 레이아웃이 중요한가요?

### SIMD 최적화
SIMD는 Single Instruction, Multiple Data의 약자로, 하나의 명령으로 여러 데이터에 대해 동일한 연산을 동시에 수행할 수 있게 해주는 CPU 기능입니다. 
  
예를 들어, 네 번의 덧셈을 각각 따로 수행하는 대신:

```python
[1+1, 2+2, 3+3, 4+4]  # 하나씩 처리
```

SIMD를 사용하면 CPU가 벡터화된 명령을 통해 이를 한 번에 처리할 수 있습니다:

```python
[1, 2, 3, 4] + [1, 2, 3, 4]  # 내부적으로 함께 계산
```

다만 SIMD는 데이터가 메모리상에 연속적으로 저장되어 있을 때만 효과적으로 동작합니다. 
데이터가 파이썬 객체처럼 여기저기 흩어져 저장되어 있으면 CPU가 여러 값을 한 번에 효율적으로 불러오거나 처리하기 어렵습니다. 
이런 이유로 Arrow 등의 라이브러리는 연속적인 메모리 레이아웃을 요구하며, 이를 통해 CPU 하드웨어의 벡터화 성능을 최대한 활용할 수 있습니다.

### 캐시 효율성
현대 CPU는 메인 메모리보다 훨씬 빠릅니다. 그래서 데이터를 메모리에서 로드하는 시간이 오히려 병목이 될 수 있습니다.
이러한 문제를 완화하기 위해 CPU는 캐시라는 작은 고속 메모리를 사용하는데, 최근에 접근한 데이터를 여기에 저장해 두었다가 다시 필요할 때 빠르게 접근할 수 있도록 합니다.

![](https://github.com/hyunwoongko/nanoRLHF/blob/main/nanorlhf/nanosets/docs/assets/cache.png?raw=true)

메모리에서 데이터를 로드할 때, CPU는 한 번에 연속된 여러 바이트를 캐시에 올립니다.
만약 데이터가 NumPy나 Arrow 배열처럼 메모리상에 연속적으로 저장되어 있다면, CPU는 한 번의 로드로 연속된 여러 원소를 캐시에 함께 올릴 수 있습니다.
그 결과, 순차적으로 처리할 때 주변 데이터가 이미 캐시에 있어 빠르게 접근할 수 있습니다.

반대로 파이썬의 PyObject 포인터 리스트처럼 데이터가 메모리 여기저기에 흩어져 있으면, 각 원소가 완전히 다른 위치에 있을 수 있습니다.
이 경우 CPU는 캐시에 있는 데이터를 재활용하지 못하고 메인 메모리에서 계속 가져와야 하며, 그 과정에서 캐시 미스가 자주 발생해 성능이 크게 저하됩니다.
이 때문에 Arrow는 모든 값을 연속적인 버퍼에 저장하여, 순차 접근 시 CPU 캐시와 메모리 프리페치의 이점을 최대한 누릴 수 있도록 설계되어 있습니다.

## 3. 왜 Validity를 Bitmap으로 저장하나요?
Validity Bitmap은 결측치 정보를 Bitmap(bitmap) 형태, 즉 바이트가 아닌 비트 단위로 저장합니다.
예를 들어, 8개의 원소에 대한 결측치 정보를 저장한다고 가정해봅시다.

| Index        | 0  | 1 | 2  | 3  | 4 | 5  | 6  | 7  |
|--------------|----|---|----|----|---|----|----|----|
| Value        | 10 | - | 30 | 40 | - | 60 | 70 | 80 |
| Validity Bit | 1  | 0 | 1  | 1  | 0 | 1  | 1  | 1  |

만약 우리가 결측치를 바이트 단위로 저장하면 총 8바이트의 용량이 필요합니다.
하지만 결측치는 0과 1로만 표현되므로 사실 비트 단위로도 저장 할 수 있습니다.
우리가 결측치 저장에 비트 단위를 사용하여 각 비트가 원소 하나의 유효성을 나타내도록 하면 동일한 데이터를 1바이트 (=8비트)로 저장할 수 있습니다.
이를 통해 결측치 정보를 8배 더 적은 용량으로 저장할 수 있습니다.

## 4. `get_validity` 구현

전체 코드는 다음과 같습니다.
```python
def get_validity(buffer: Buffer, i: int) -> bool:
    byte, bit = divmod(i, 8)
    b = buffer.data[byte]
    mask = (1 << bit)
    check = b & mask
    return check != 0
```

`i`는 bitmap에서 조회하고자 하는 원소의 인덱스입니다. 예를 들어 13번째 원소의 유효성을 조회한다고 해봅시다.

우리는 먼저 13이 몇번째 바이트의 몇번째 비트에 해당하는지 계산해야 합니다. 이를 `divmod` 함수를 사용해 쉽게 구할 수 있습니다.

```python
byte, bit = divmod(13, 8)
# byte = 1, bit = 5
```
그 다음 bitmap의 해당 바이트(1번 바이트)를 읽어옵니다.

```python
b = bitmap.data[byte]
# b = 0b00101101  # 예시 값
```

이제 우리가 관심있는 비트 위치에 해당하는 마스크를 생성합니다.

```python
mask = (1 << bit)  # 1을 5번 왼쪽으로 시프트
# mask = 0b00100000
```

AND 연산을 통해 해당 비트가 1인지 0인지 확인합니다.

```python
check = b & mask

# b     = 0b00101101
# mask  = 0b00100000
# ------------------
# check = 0b00100000
```

만약 해당 비트가 결측치라면 AND 연산에 의해 모든 비트는 0이 될 것이고 결과적으로 `check`는 `0b00000000`, 즉 0이 됩니다.
하지만 유효한 값이라면 적어도 하나의 비트는 1이 될 것이고 결과적으로 `check`는 0이 아닌 어떤 값이 됩니다.

따라서 `check`가 0인지 아닌지 확인하여 유효성을 반환합니다.

```python
return check != 0  # True (유효한 값)
```

## 5. `set_validity` 구현
전체 코드는 다음과 같습니다.
```python
def set_validity(buffer: Buffer, i: int, valid: bool):
    byte, bit = divmod(i, 8)
    b = buffer.data[byte]

    if valid:
        mask = (1 << bit)
        packed = b | mask
    else:
        mask = ~(1 << bit) & 0xFF
        packed = b & mask

    buffer.data[byte] = packed
```

`i`는 bitmap에서 설정하고자 하는 원소의 인덱스이고, `valid`는 해당 원소를 유효하게 설정할지 (True) 결측치로 설정할지 (False)를 나타냅니다.

만약 13번째 원소를 결측치로 설정한다고 해봅시다.
그러면 우리는 13번째 원소가 몇번째 바이트의 몇번째 비트에 해당하는지 계산해야 합니다. 결측치 조회에서 사용한 것과 동일한 방법을 사용합니다.

```python
byte, bit = divmod(13, 8)
# byte = 1, bit = 5
```

그 다음 bitmap의 해당 바이트(1번 바이트)를 읽어옵니다.

```python
b = bitmap.data[byte]
# b = 0b00001101  # 예시 값
```

만약 설정하고 싶은 `valid`가 True라면 다음과 같이 OR 연산을 이용해 비트를 설정합니다.

```python
mask = (1 << bit)  # 1을 5번 왼쪽으로 시프트
packed = b | mask

# b     = 0b00001101
# mask  = 0b00100000
# ------------------
# packed= 0b00101101  # 5번째 비트가 1로 설정됨
```

반대로 `valid`가 False라면 다음과 같이 AND 연산을 이용해 비트를 클리어합니다.

```python
mask = ~(1 << bit)  # 1을 5번 왼쪽으로 시프트 후 비트 반전
packed = b & mask

# b      = 0b00101101
# mask   = 0b11011111
# -------------------
# packed = 0b00001101  # 5번째 비트가 0으로 설정됨
```

이때 주의할 점은 비트 반전 연산(~)이 원치 않는 영역까지 반전시킬 수 있다는 것입니다. Numpy처럼 C언어를 기반으로 하면 정수형이 고정된 비트 수(예: 8비트, 16비트, 32비트 등)를 가지기 때문에 이런 문제가 발생하지 않지만,
우리는 Bitmap을 Python에서 다루고 있기 때문에 정수형을 특정 비트 수로 제한 할 수 없습니다.

예를 들어, 16비트 정수에서 5번째 비트를 반전시키면 `0b1111111111011111`이 됩니다. 우리는 하위 8비트만 AND 연산에 사용하고 싶기 때문에 사용하지 않는 앞 8개의 비트는 
연산에 쓰이지 못하도록 제거해야 합니다. 참고로 0xFF는 하위 8개의 비트만 1로 설정된 값인데 이것과 어떤 값을 AND 연산하면 하위 8비트만 남게 됩니다.

```python
mask = ~(1 << bit)
mask = mask & 0xFF  # 하위 8비트만 남김

# mask = 0b1111111111011111
# 0xFF = 0b0000000011111111
# -------------------------
# mask = 0b0000000011011111
```

이를 반영해 AND 연산을 수행해 값을 계산합니다.

```python
mask = ~(1 << bit) & 0xFF
packed = b & mask

# b      = 0b0000000000101101
# mask   = 0b0000000011011111
# ---------------------------
# packed = 0b0000000000001101  # 5번째 비트가 0으로 설정됨
```

마지막으로 수정된 packed 값을 bitmap에 다시 저장합니다.

```python
bitmap.data[byte] = packed
# bitmap.data[1] = 0b0000000000001101
```

## 6. `slice` 구현
전체 코드는 다음과 같습니다.
```python
import math


def slice(buffer: Buffer, offset: int, i: int, slice_length: int) -> Bitmap:
    """i부터 i + slice_length 까지의 비트를 참조하는 새로운 Bitmap 뷰를 생성합니다."""

    abs_bit_position = offset + i
    byte, bit = divmod(abs_bit_position, 8)
    needed_bytes = math.ceil((bit + slice_length) / 8)
    sliced_buffer = buffer.data[byte:byte + needed_bytes]
    return Bitmap(slice_length, sliced_buffer, offset=bit)
```

Bitmap slicing 구현의 핵심은 원본 Bitmap을 복사하여 새로운 Bitmap을 만드는 것이 아니라,
Zero-copy의 철학대로 원본 Bitmap의 특정 구간을 가리키는 새로운 뷰(view)를 만드는 것입니다.

따라서 원본 Bitmap은 동일하게 유지하면서 
우리가 지금 이 Bitmap의 어디부터 어디까지를 볼 것인지를 메타데이터(offset)로 관리하도록 구현합니다.

이러한 방식으로 slicing을 구현하기 위해서는 absolute bit position이라는 개념이 필요합니다.
absolute bit position은 논리적인 인덱스 `i`를 실제 버퍼 상의 비트 위치로 변환한 값입니다. 
이 문서에서는 이를 다음과 같이 정의합니다.

- logical index: 사용자 입장에서 보이는 인덱스 `i` (0부터 시작)
- offset: 이 Bitmap view는 몇 번째 비트에서부터 읽어야 하는지 나타내는 값
- absolute bit position: `abs_bit = offset + i`

즉, 같은 Bitmap이라도 이미 어디선가 slicing 되었다면 0번부터 읽지 않아야 합니다.
이를 offset이라는 값에 담아두고, logical index `i`에 offset을 더한 값을 absolute bit position으로 사용합니다.
예를 들어 `offset = 3`이고 유저가 원하는 logical index가 `i = 2`라면 우리는 원본 비트맵의 2번째 원소를 조회하는 것이 아니라
`abs_bit_position = 3 + 2 = 5`번째 비트를 조회해야 합니다.

이제 코드를 살펴봅시다.

위에서 언급한대로 logical index `i`를 absolute bit position으로 변환합니다.

```python
abs_bit_position = offset + i
```

absolute bit position을 사용해 새로운 뷰의 시작 위치를 계산합니다.
만약 `i`가 2이고 `offset`이 3이라면 absolute bit position은 5가 되고,
byte는 0, bit는 5가 됩니다.

```python
byte, bit = divmod(abs_bit_position, 8)
```

이 뷰를 표현하기 위해 몇 바이트가 필요한지 계산합니다.
왜 `bit + slice_length`를 8로 나눈 후 올림을 하는지 주의깊게 살펴봅시다.

만약 `bit`가 3이고 `slice_length`가 10이라면,
- bit + slice_length = 3 + 10 = 13
- 13 / 8 = 1.625
- 올림(1.625) = 2

총 2바이트를 사용해야 합니다.

왜냐하면 buffer에서는 첫 바이트의 3번째 비트부터 시작해 10개의 비트를 읽으려면, 첫 바이트의 3번째 비트부터 끝까지 5개 비트를 읽고, 
나머지 5개 비트는 2번째 바이트에서 읽어야 하기 때문입니다.

```python
needed_bytes = math.ceil((bit + slice_length) / 8)
```

원본 버퍼에서 필요한 바이트 구간을 슬라이스하여 새로운 버퍼 뷰를 생성합니다.
참고로 Buffer는 내부적으로 `memoryview`를 사용하기 때문에 이 과정에서 새로운 복사가 일어나지 않습니다.

```python
sliced_buffer = buffer.data[byte:byte + needed_bytes]
```

마지막으로 새로운 Bitmap 뷰를 생성하여 반환합니다.

```python
return Bitmap(slice_length, sliced_buffer, offset=bit)
```

그리고 기존의 `get_validity` 및 `set_validity` 함수들은 이 새로운 offset 메타데이터를 사용하도록 수정되어야 합니다.
그러면 slicing 된 Bitmap 뷰에서도 정확한 위치의 비트를 읽고 쓸 수 있게 됩니다.

```python
def get_validity(buffer: Buffer, offset: int, i: int) -> bool:
    abs_bit_position = offset + i  # <--- offset 반영
    byte, bit = divmod(abs_bit_position, 8)
    b = buffer.data[byte]
    mask = (1 << bit)
    check = b & mask
    return check != 0


def set_validity(buffer: Buffer, offset: int, i: int, valid: bool):
    abs_bit_position = offset + i  # <--- offset 반영
    byte, bit = divmod(abs_bit_position, 8)
    b = buffer.data[byte]

    if valid:
        mask = (1 << bit)
        packed = b | mask
    else:
        mask = ~(1 << bit) & 0xFF
        packed = b & mask

    buffer.data[byte] = packed
```

## Validity Bitmap: 3줄 요약
- Validity Bitmap은 결측치 정보를 비트 단위로 저장하여 메모리 사용량을 줄이고 연속된 메모리 레이아웃을 유지합니다.
- 연속된 메모리 레이아웃은 SIMD 최적화와 캐시 효율성을 높여 성능 향상에 기여합니다.
- Bitmap은 비트 단위로 조희, 저장, 슬라이싱 기능을 구현하여 효율적으로 결측치를 관리할 수 있습니다.