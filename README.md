<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="./static/icon.png" alt="Project logo" ></a>
 <br>

</p>

<h3 align="center">Programmers</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/da-huin/hadoop-tutorial.svg)](https://github.com/da-huin/hadoop-tutorial/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/da-huin/hadoop-tutorial.svg)](https://github.com/da-huin/hadoop-tutorial/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>


# Programmers Level 1

## 크레인 인형뽑기 게임

https://programmers.co.kr/learn/courses/30/lessons/64061

### 아이디어

* 인형을 가져와 Stack 에 넣어서 이전과 같은 모양이면 stack.pop() 으로 마지막을 꺼낸다. 

* 사이즈가 매우 커진다면, for 문으로 한 줄을 전부 읽는 것이 아니고 큐를 사용하면 될 것 같다.


### 코드

```python

def solution(board, moves):
    n = len(board)
    stack = []
    answer = 0
    for i in moves:
        for j in range(n):
            doll = board[j][i - 1]
            if doll == 0:
                continue
            
            board[j][i - 1] = 0

            if len(stack) and stack[-1] == doll:
                stack.pop()
                answer += 2
            else:
                stack.append(doll)
            break
    
    return answer
        
```

---

## 완주하지 못한 선수

https://programmers.co.kr/learn/courses/30/lessons/42576

### 아이디어

* 단 한명의 선수 빼고는 모두 완주했다고 한다.
* 참가자의 모든 해쉬를 더하고 완주자의 모든 해쉬를 빼면 남은 값은 완주하지 못한 사람의 해쉬가 된다.

### 코드

```python

def solution(participant, completion):
    running = {}
    hash_sum = 0

    for person in participant:
        running[hash(person)] = person
        hash_sum += hash(person)

    for person in completion:
        hash_sum -= hash(person)
    
    return running[hash_sum]

```

---

## 두 개 뽑아서 더하기

https://programmers.co.kr/learn/courses/30/lessons/68644

### 아이디어

* 모든 경우의수를 위해 for문을 중첩으로 돌린다.

### 코드

```python

def solution(numbers):
    return sorted(list(set([numbers[i] + numbers[j] for i in range(len(numbers)) for j in range(len(numbers)) if i != j])))

```

---

## 모의고사

https://programmers.co.kr/learn/courses/30/lessons/42840

### 아이디어

* `index % len(chocies)` 로 하나씩 비교한다.

### 코드

```python

def solution(answers):
    scores = []
    for choices in [[1,2,3,4,5], [2,1,2,3,2,4,2,5], [3,3,1,1,2,2,4,4,5,5]]:
        scores.append([True for index, answer in enumerate(answers) if answer == choices[index % len(choices)]])
        
    max_score = max(scores)
    
    return sorted([index + 1 for index, score in enumerate(scores) if max_score == score])

```

---

## 체육복

https://programmers.co.kr/learn/courses/30/lessons/42862

### 아이디어

* 자기 자신이 잃어버릴 수도 있으므로 그 경우는 미리 제거해야 한다.
* 왼쪽이 있으면 먼저 제거하고 그 이후에 오른쪽을 제거하면 최대로 빌려 줄 수 있다.

### 코드

```python
def solution(n, lost, reserve):
    answer = 0
    _lost = set([l for l in lost if l not in reserve])
    _reserve = set([r for r in reserve if r not in lost])
    lost = _lost
    reserve = _reserve

    answer = n
    for r in reserve:
        front = r - 1        
        back = r + 1

        if front in lost:
            lost.remove(front)

        elif back in lost:
            lost.remove(back)

    answer -= len(lost)
    
    return answer
```

---

## K 번째 수

https://programmers.co.kr/learn/courses/30/lessons/42748

### 아이디어

* 그냥 잘라서 인덱싱하면 된다.

### 코드

```python
def solution(array, commands):
    return [sorted(array[i-1:j])[k-1] for i, j, k in commands]
```

---

## 2016년

https://programmers.co.kr/learn/courses/30/lessons/12901

### 아이디어

* datetime 으로 날짜를 요일을 가져오면 된다.

### 코드

```python

import datetime
def solution(a, b):
    weekdays = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    weekday = datetime.datetime.strptime(f"2016-{str(a).zfill(2)}-{str(b).zfill(2)}", "%Y-%m-%d").weekday()
    return weekdays[weekday]

```

---

## 가운데 글자 가져오기

https://programmers.co.kr/learn/courses/30/lessons/12903

### 아이디어

* 짝수, 홀수 나누어서 반환하면 된다.

### 코드

```python
def solution(s):
    answer = ''
    m = int(len(s) / 2)
    
    if len(s) % 2:
        answer = s[m]
    else:
        answer = s[m - 1] + s[m]
    
    return answer
```

---

## 같은 숫자는 싫어

https://programmers.co.kr/learn/courses/30/lessons/12906

### 아이디어

* 현재 숫자가 다음 숫자와 같지 않으면 정답에 추가한다.

### 코드

```python
def solution(arr):
    return [n for i, n in enumerate(arr) if not (i + 1 < len(arr) and n == arr[i + 1])]
```


---

## 나누어 떨어지는 배열

https://programmers.co.kr/learn/courses/30/lessons/12910

### 아이디어

* 나누어 떨어지면 (`n % divisor`) 정답에 추가한다.

### 코드

```python
def solution(arr, divisor):
    answer = sorted([n for n in arr if not (n % divisor)])
    return answer if len(answer) else [-1]
```



---

## 두 정수 사이의 합

https://programmers.co.kr/learn/courses/30/lessons/12912

### 아이디어

* 작은 수 부터 큰 수 까지의 합을 반환한다.

### 코드

```python
def solution(a, b):
    return sum(list(range(min(a, b), max(a, b) + 1)))
```

---

## 문자열 내 마음대로 정렬하기

https://programmers.co.kr/learn/courses/30/lessons/12915

### 아이디어

* tuple 을 이용하여 string 의 n 번째를 기준으로 정렬한다.

### 코드

```python
def solution(strings, n):
    return [x[1] for x in sorted([(string[n], string) for string in strings])]
```

---

## 문자열 내 p와 y의 개수

https://programmers.co.kr/learn/courses/30/lessons/12916

### 아이디어

* counter 를 이용하여 갯수를 센다.

### 코드

```python
from collections import Counter
def solution(s):
    c = Counter(s.lower())
    return c.get("y") == c.get("p")
```

---

## 문자열 내림차순으로 배열하기

https://programmers.co.kr/learn/courses/30/lessons/12917

### 아이디어

* sorted 로 정렬한다.

### 코드

```python
def solution(s):
    return "".join(sorted(s, reverse=True))
```
---

## 문자열 다루기 기본

https://programmers.co.kr/learn/courses/30/lessons/12918

### 아이디어

* isdigit 함수로 숫자인지 확인한다.

### 코드

```python
def solution(s):
    return (len(s) == 4 or len(s) == 6) and s.isdigit()
```
---

## 서울에서 김서방 찾기

https://programmers.co.kr/learn/courses/30/lessons/12919

### 아이디어

* index 를 사용하여 위치를 찾는다.

### 코드

```python
def solution(seoul):
    return f"김서방은 {seoul.index('Kim')}에 있다"
```

---

## 소수 찾기

https://programmers.co.kr/learn/courses/30/lessons/12921

### 아이디어

* 에라스토테네스의 체를 이용한다.

### 코드

```python
def solution(n):
    n = n + 1
    sieve = [True] * n
    
    m = int(n ** 0.5)
    for i in range(2, m + 1):
        if sieve[i]:
            for j in range(i * 2, n, i):
                sieve[j] = False
                
    return len([i for i in range(2, n) if sieve[i]])
```

---

## 수박수박수박수박수박수?

https://programmers.co.kr/learn/courses/30/lessons/12922

### 아이디어

* (수박 * n)개 만큼 만들어두고 n개만큼 자른다.

### 코드

```python
def solution(n):
    return ("수박" * n)[:n]
```


---

## 문자열을 정수로 바꾸기

https://programmers.co.kr/learn/courses/30/lessons/12925

### 아이디어

* int 를 사용한다.

### 코드

```python
def solution(s):
    return int(s)
```


---

## 시저 암호

https://programmers.co.kr/learn/courses/30/lessons/12926

### 아이디어

* ord 를 사용하여 인덱싱을 한다.

### 코드

```python
import string
def solution(s, n):
    answer = ''
        
    for ch in s:
        if ch in string.ascii_lowercase:
            caesar = string.ascii_lowercase[(ord(ch) - ord("a") + n) % len(string.ascii_lowercase)]
        elif ch in string.ascii_uppercase:
            caesar = string.ascii_uppercase[(ord(ch) - ord("A") + n) % len(string.ascii_uppercase)]
        else:
            caesar = ch
            
        answer += caesar

    return answer
```

---

## 약수의 합

https://programmers.co.kr/learn/courses/30/lessons/12928

### 아이디어

* 약수를 구하는 공식이 있다.

### 코드

```python
def solution(n):
    d = [] 
    for m in range(1, int(n ** 0.5) + 1):
        if n % m == 0:
            d.extend([m, n//m])
    return sum(set(d))
```

---

## 이상한 문자 만들기

https://programmers.co.kr/learn/courses/30/lessons/12930

### 아이디어

* map 을 이용하여 함수를 전체에 적용시킨다.

### 코드

```python
def solution(s):
    return " ".join(map(lambda x: "".join([w.lower() if i % 2 else w.upper() for i, w in enumerate(x)]), s.split(" ")))
```

---

## 자릿수 더하기

https://programmers.co.kr/learn/courses/30/lessons/12931

### 아이디어

* string 으로 만들어서 자릿수를 더한다.

### 코드

```python
def solution(n):
    return sum([int(n) for n in list(str(n))])
```

---

## 자연수를 뒤집어 배열로 만들기

https://programmers.co.kr/learn/courses/30/lessons/12932

### 아이디어

* reversed 로 뒤집는다.

### 코드

```python
def solution(n):
    return [int(n) for n in reversed(list(str(n)))]
```

---

## 정수 내림차순으로 배치하기

https://programmers.co.kr/learn/courses/30/lessons/12933

### 아이디어

* sorted 로 내림차순으로 배치한다.

### 코드

```python
def solution(n):
    return int("".join([i for i in sorted(list(str(n)), reverse=True)]))
```

---

## 정수 제곱근 판별

https://programmers.co.kr/learn/courses/30/lessons/12934

### 아이디어

* `**2` 로 제곱인지 판단한다.

### 코드

```python
def solution(n):
    root = n ** 0.5
    return (root + 1)**2 if float(root) == int(root) else -1
```

---

## 제일 작은 수 제거하기

https://programmers.co.kr/learn/courses/30/lessons/12935

### 아이디어

* min 과 remove 로 최솟값을 제거한다.

### 코드

```python
def solution(arr):
    arr.remove(min(arr))
    return arr if len(arr) else [-1]
```

---

## 짝수와 홀수

https://programmers.co.kr/learn/courses/30/lessons/12937

### 아이디어

* `% 2` 로 짝수인지 홀수인지 판단한다.

### 코드

```python
def solution(num):
    return "Odd" if num % 2 else "Even"
```

---

## [카카오 인턴] 키패드 누르기

https://programmers.co.kr/learn/courses/30/lessons/67256

### 아이디어

* 키패드의 위치를 기억하고, 그에 따라서 실행을 한다.

### 코드

```python
def get_distance(thumb, y):
    tx, ty = thumb
    return abs(ty - y) + int(tx != 1)

def solution(numbers, hand):
    answer = ""

    keypad = [
        [1, 4, 7, "*"],
        [2, 5, 8, 0],
        [3, 6, 9, "#"]]
    
    thumb = {
        "L": [0, 3],
        "R": [2, 3]
    }
    
    for n in numbers:
        pos = ""
        x = -1
        y = -1
        
        for i in range(3):
            if n not in keypad[i]:
                continue

            x = i
            y = keypad[i].index(n)
        
        if n in keypad[0]:
            pos = "L"
        elif n in keypad[2]:
            pos = "R"
        else:
            ld = get_distance(thumb["L"], y)
            rd = get_distance(thumb["R"], y)
            if ld == rd:
                pos = "R" if hand == "right" else "L"
            elif ld > rd:
                pos = "R"
            else:
                pos = "L"

        thumb[pos][0] = x
        thumb[pos][1] = y
        
        answer += pos

    return answer
```

---

## 최대공약수와 최소공배수

https://programmers.co.kr/learn/courses/30/lessons/12940

### 아이디어

* gcd(greatest common divisor) 함수를 사용한다.

### 코드

```python
import math

def solution(n, m):
    return [math.gcd(n, m), n * m / math.gcd(n, m)]
```

---

## 콜라크 추측

https://programmers.co.kr/learn/courses/30/lessons/12943

### 아이디어

* 문제 그대로 구현하면 된다.

### 코드

```python
def solution(num):
    for i in range(500):

        if num == 1:
            break
            
        if num % 2:
            num = num * 3 + 1
        else:
            num = num / 2

    return i if i != 499 else -1
```

---

## 평균 구하기

https://programmers.co.kr/learn/courses/30/lessons/12944

### 아이디어

* 평균 = 전체 / 갯수

### 코드

```python
def solution(arr):
    return sum(arr) / len(arr)
```

---

## 하샤드 수

https://programmers.co.kr/learn/courses/30/lessons/12947

### 아이디어

* 설명하는 그대로 구현한다.

### 코드

```python
def solution(x):
    return x % sum([int(n) for n in list(str(x))]) == 0
```

---

## 핸드폰 번호 가리기

https://programmers.co.kr/learn/courses/30/lessons/12948

### 아이디어

* `- 인덱싱`을 이용해 별표로 만든다.

### 코드

```python
def solution(phone_number):
    return (len(phone_number[:-4]) * "*") + phone_number[-4:]
```

---

## 행렬의 덧셈

https://programmers.co.kr/learn/courses/30/lessons/12950

### 아이디어

* for 문을 두 개 사용하여 더한다.

### 코드

```python
def solution(arr1, arr2):
    return [[arr2[i][j] + n for j, n in enumerate(p1)] for i, p1 in enumerate(arr1)]
```

---

## x만큼 간격이 있는 n개의 숫자

https://programmers.co.kr/learn/courses/30/lessons/12954

### 아이디어

* 그대로 구현한다.

### 코드

```python
def solution(x, n):
    return [x*i for i in range(1, n + 1)]
```

---

## 직사각형 별찍기

https://programmers.co.kr/learn/courses/30/lessons/12969

### 아이디어

* 그대로 구현한다.

### 코드

```python
n, m = map(int, input().strip().split(' '))

for _ in range(m):
    print("*" * n)
```

---

## 예산

https://programmers.co.kr/learn/courses/30/lessons/12982

### 아이디어

* 사용해야하는 금액을 작은 순으로 정렬해 예산이 초과하기 전까지 사용한다.

### 코드

```python
def solution(d, budget):
    answer = 0
    
    for price in sorted(d):
        if budget - price >= 0:
            budget = budget - price
            answer += 1
        else:
            break
            
    return answer
```

---

## [1차] 비밀지도

https://programmers.co.kr/learn/courses/30/lessons/17681

### 아이디어

* 비트 연산자 참조 (https://wikidocs.net/1161)

### 코드

```python
def solution(n, arr1, arr2):
    return ["".join(map(lambda x: "#" if int(x) else " ", bin(i | j)[2:].zfill(n))) for i, j in zip(arr1, arr2)]
```

---

## 실패율

https://programmers.co.kr/learn/courses/30/lessons/42889

### 아이디어

* 문제가 이해하기 힘들 수도 있으므로 잘 읽는다.
* Multi Sorting 을 활용하여 정럴한다.

### 코드

```python
def get_failrate(stage, stages):
    reachers = 0
    challengers = 0
    
    for x in stages:
        if x == stage:
            challengers += 1
        if x >= stage:
            reachers += 1
    
    return challengers / reachers if reachers != 0 else -1
    
def solution(N, stages):
    failrates = [(stage, get_failrate(stage, stages)) for stage in range(1, N + 1)]

    return [x[0] for x in sorted(failrates, key=lambda x: (-x[1], x[0]))]
```

---

## 다트 게임

https://programmers.co.kr/learn/courses/30/lessons/17682

### 아이디어

* 그룹화를 사용하여 가져오면 직접 분리하지 않아도 사용 할 수 있다.

  ```python
  re.compile('(\d+)([SDT])([*#]?)')
  ```

### 코드

```python
import re
def solution(dartResult):
    p = re.compile('(\d+)([SDT])([*#]?)')
    print(p.findall(dartResult))
    return

    table = list(zip(re.findall(r"\d+", dartResult), re.split(r"\d+", dartResult)[1:]))
    
    
    scores = []
    
    for n, assist in table:
        n = int(n)
        option = assist[1] if len(assist) == 2 else ""
        bonus = {"S":1, "D":2, "T":3}[assist[0]]
        
        score = n ** bonus
        
        if option == "*":
            if len(scores) > 0:
                scores[-1] *= 2
            score *= 2
        
        elif option == "#":
            score *= -1
            
        scores.append(score)
    
    return sum(scores)
```

# Programmers Level 2

## 124 나라의 숫자

https://programmers.co.kr/learn/courses/30/lessons/12899

### 아이디어

* 진법을 변환하는 방법을 찾아서 푸는 것이 편하다.

### 코드

```python
def solution(n):
    law = '124'
    answer=""

    while n > 0:
        n -= 1
        answer = law[n % 3] + answer
        n = n // 3
        
    return answer
```

---

## 주식가격

https://programmers.co.kr/learn/courses/30/lessons/42584

### 아이디어

* 각자 몇 초 동안 유지되었는지 확인해야하므로 for 문을 돌린다.
* 각자의 유지 기간을 계산한다.
* 바로 떨어져도 1초간 유지되는 것이므로 else 에서도 +1을 해준다.

### 코드

```python
def solution(prices):
    n = len(prices)
    answer = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if prices[i] <= prices[j]:
                answer[i] += 1
            else:
                answer[i] += 1
                break
        
    return answer
```

---

## 다리를 지나는 트럭

https://programmers.co.kr/learn/courses/30/lessons/42583

### 아이디어

* 1초마다 이동하는 방식은 느리므로, 계산하지 않아도 되는 시간은 바로 없애버린다.

### 코드

```python
def select(weight, bridge_map, truck_weights, on_bridge_weight):
    if len(truck_weights):
        
        truck = truck_weights[-1]

        if weight - on_bridge_weight - truck >= 0:
            return truck_weights.pop()

    return 0

def solution(bridge_length, weight, truck_weights):
    on_bridge_weight = 0
    answer = 0
    bridge_map = [0] * bridge_length
    move = lambda x, j: [0] * j + x[:-j]
    truck_weights = list(reversed(truck_weights))
    
    unselect = False
    while True:

        last_truck = bridge_map[-1]
        on_bridge_weight -= last_truck
                
        bridge_map = move(bridge_map, 1)
        answer += 1

        truck = select(weight, bridge_map, truck_weights, on_bridge_weight)
                    
        on_bridge_weight += truck
        bridge_map[0] = truck

        if len(truck_weights) == 0 and sum(bridge_map) == 0:
            break

        if truck == 0:
            n = 0
            for index, block in enumerate(reversed(bridge_map)):
                if block != 0:
                    n = index
                    break
                    
            if n != 0:
                bridge_map = move(bridge_map, n)
            answer += n
        
    return answer
```

---

## 기능개발

https://programmers.co.kr/learn/courses/30/lessons/42586

### 아이디어

* 문제 그대로 구현한다.

### 코드

```python
import math
def solution(progresses, speeds):
    answer = []
    stage = list(map(lambda x: math.ceil((100 - x[0]) / x[1]), zip(progresses, speeds)))
    
    while True:
        least = min([n for n in stage if n > 0])
        stage = [day - least for day in stage]
        
        count = 0
        for day in stage:
            if day <= 0:
                count += 1
            else:
                break
                
        if count != 0:
            answer.append(count)
            stage = stage[count:]

        if len(stage) == 0:
            break


    return answer
```

---

## 프린터

https://programmers.co.kr/learn/courses/30/lessons/42587

### 아이디어

* 큐를 사용하여 해결한다.
* 설명 그대로 따라가며 해결하는 편이 낫다.

### 코드

```python
from queue import Queue
def solution(priorities, location):
    answer = 0
    q = Queue()
    
    list(map(lambda x: q.put(x), list(zip(priorities, range(len(priorities))))))
    
    while len(q.queue):
        hp, _ = max(list(q.queue))
        p, i = q.get()

        if p < hp:
            q.put((p, i))
        else:
            answer += 1
            if i == location:
                break

    return answer
```

---

## 스킬트리

https://programmers.co.kr/learn/courses/30/lessons/49993

### 아이디어

* 설명을 그대로 따라가면서 구현한다.

### 코드

```python
import re
def solution(skill, skill_trees):
    answer = 0

    str_trees = " ".join(skill_trees)
    for i, k in enumerate(skill):
        str_trees = str_trees.replace(k, str(i))

    for tree in str_trees.split(" "):
        nums = re.findall("\d", tree)

        if nums:
            check = 0
            for n in nums:
                if check == int(n):
                    check += 1
                else:
                    answer -= 1
                    break
        answer += 1
    
    return answer
```

---

## 멀쩡한 사각형

https://programmers.co.kr/learn/courses/30/lessons/62048

### 아이디어

* 이 문제를 보며 직접 최대공약수로 푼다고 떠올리기는 힘든 것 같다.

### 코드

```python
import math
def solution(w,h):
    return ((h * w) - (h + w - math.gcd(h, w)))
```

---

## 문자열 압축

https://programmers.co.kr/learn/courses/30/lessons/60057

### 아이디어

* 정규식으로 매치시켜 해결한다.

### 코드

```python
import re
def solution(s):
    answer = 0
    shorts = []
    
    for size in range(1, int(len(s) / 2) + 1):
        short = ""
        worker = s
        while len(worker) - size > 0:
            word = worker[:size]

            count = int(len(re.match("^(" + word + ")+", worker)[0]) / size)
            short += f"{count if count > 1 else ''}{word}"
            worker = worker[int(size * count):]
        
        short += worker
        shorts.append(short)

    if len(shorts) == 0:
        shorts.append(s)

    return min([len(short) for short in shorts if short != ""])
```

---

## 삼각 달팽이

https://programmers.co.kr/learn/courses/30/lessons/68645

### 아이디어

* `status` 라는 변수를 사용하여 아래로 내려갈 때, 오른쪽으로 갈 때, 위로 갈 때를 각각 설정하여 실제로 숫자를 넣는다.

### 코드

```python
def can_move(rows, i, j):
    if i < 0 or j < 0 or i >= len(rows) or j >= len(rows[i]) or rows[i][j] != 0:
        return False

    return True

def solution(n):
    answer = []
    rows = [[0] * (i + 1) for i in range(n)]
    num_count = 1 + sum([i + 2 for i in range(n - 1)])
    n = len(rows)
    i = 0
    j = 0
    status = 0

    DELTAS = [1, 0], [0, 1], [-1, -1]

    for num in range(1, num_count + 1):

        rows[i][j] = num
        di, dj = DELTAS[status]

        if not can_move(rows, i + di, j + dj):
            status = (status + 1) % 3
            di, dj = DELTAS[status]
        
        i += di
        j += dj
    
    for row in rows:
        answer.extend(row)
    return answer
```

---

## 큰 수 만들기

https://programmers.co.kr/learn/courses/30/lessons/42883

### 아이디어

* 문제를 따라가면서 그대로 구현한다.

### 코드

```python
def solution(number, k):
    answer = ''
    nums = list(number)
    
    size = len(nums) - k
    head = len(nums)
    nums = [int(n) for n in nums]
    
    bis = 0
    for i in range(size):
        candidate = nums[bis:head -(size-i) + 1]
        bn = -1
        bi = -1
        for i, n in enumerate(candidate):
            if bn < n:
                bn = n
                bi = i
                if n == 9:
                    break

        bis += bi + 1
        answer += str(bn)

    return answer
```

---

## 더 맵게

https://programmers.co.kr/learn/courses/30/lessons/42626

### 아이디어

* heap 을 사용하여 스코빌 순으로 처리한다.

### 코드

```python
import heapq
def solution(scoville, K):
    heap = []
    
    list(map(lambda v: heapq.heappush(heap, v), scoville))
    
    count = 0
    success = False
    while len(heap):
        v1 = heapq.heappop(heap)
        
        if v1 >= K:
            success = True
            break
        
        if len(heap):
            v2 = heapq.heappop(heap)
            heapq.heappush(heap, v1 + (v2 * 2))
            count += 1
        
    return count if success else -1
```

---

## 괄호 변환

https://programmers.co.kr/learn/courses/30/lessons/60058

### 아이디어

* 설명대로 따라가며 만든다.

### 코드

```python
def get_balanced(w):
    lc = 0
    rc = 0
    for o in w:
        if o == "(":
            lc += 1
        else:
            rc += 1

        if lc == rc:
            break
    
    u = "".join(w[:lc + rc])
    v = "".join(w[lc + rc:])
    
    return u, v

def is_right(w):
    check = 0
    for o in w:
        if o == "(":
            check += 1

        elif o == ")":
            if check == 0:
                break
            check -= 1
    else:
        return True
    
    return False
    
def work(w):
    if w == "":
        return ""

    u, v = get_balanced(w)
    
    if is_right(u):
        return u + work(v)

    e = f"({work(v)})" + "".join([")" if o == "(" else "(" for o in u[1:-1]])
    
    return e


def solution(p):
    return work(p)
```

---

## 소수 찾기

https://programmers.co.kr/learn/courses/30/lessons/42839

### 아이디어

* 에라스토테네스의 체를 사용하여 소수를 확인한다.

### 코드

```python
from itertools import permutations
from itertools import chain

def is_prime(n):
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            break
    else:
        return True
    return False

def solution(numbers):
    combos = [list(permutations(numbers, i)) for i in range(1, len(numbers) + 1)]
    
    combos = list(set([int("".join(p)) for p in list(chain(*combos))]))
    
    combos = list(filter(lambda x: x > 1, combos))
    
    primes = [n for n in combos if is_prime(n)]
    return len(primes)
```

---

## 가장 큰 수

https://programmers.co.kr/learn/courses/30/lessons/42746

### 아이디어

* 커스텀 정렬 (cmp_to_key) 을 사용하여 정렬한다.

### 코드

**첫 번째 방법**

```python
from functools import cmp_to_key
def sorting(a, b):
    la = len(a)
    lb = len(b)

    if la != lb:

        short = a if la < lb else b
        long = a if la > lb else b

        if long[:len(short)] == short:
            corr = -1 if la > lb else 1
            r = 1 if short + long > long + short else -1

            return r * corr
    if a < b:
        return -1
    else:
        return 1
    
def solution(numbers):
    
    sn = [str(n) for n in numbers]
    
    return str(int("".join(sorted(sn, reverse=True, key=cmp_to_key(sorting)))))

```

**두 번째 방법**

* 10 이 넘어가면 정렬이 원하는 대로 되지 않으므로 `* 3` 을 해준다.
  * `* 3`을 해주는 이유는 제한 사항에 원소가 1000 이하라고 적혀져 있기 때문이다.

```python
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x: x*3, reverse=True)
    return str(int(''.join(numbers)))
```

---

## 조이스틱

https://programmers.co.kr/learn/courses/30/lessons/42860

### 아이디어

* 문제를 따라가면서 천천히 구현한다.

### 코드

```python
def get_character_distance(target):
    distances = list(range(0, 13)) + [13] + list(reversed(range(1, 13)))
    return distances[ord(target) - ord('A')]
    
def get_near(head, fields):
    a = [(head + i, v) for i, v in enumerate(fields[head:])]
    b = [(i, v) for i, v in enumerate(fields[:head])]
    exclusive_fields = a + b

    half = int(len(exclusive_fields) / 2)
    corr = 1 if len(exclusive_fields) % 2 != 0 else 0
    distances_map = list(range(0, half + corr)) + list(reversed(range(1, half + 1)))
    distances = []
    distances.extend([(distances_map[ei], p[0]) for ei, p in enumerate(exclusive_fields) if p[1] != 0])
    return min(distances)
        
def solution(name):
    total_distance = 0
    
    fields = [get_character_distance(target) for target in name]
    
    head = 0
    while True:
        total_distance += fields[head]
        
        fields[head] = 0
            
        if not any(fields):
            break

        distance, index = get_near(head, fields)
        
        target = index

        total_distance += distance
        
        head = target
            
    return total_distance
```

---

## H-Index

https://programmers.co.kr/learn/courses/30/lessons/42747

### 아이디어

* 문제 그대로 구현한다.

### 코드

```python
def solution(citations):
    citations.sort(reverse=True)
    h = 0
    for i, q in enumerate(citations):
        nh = min(i + 1, q)
        if h > nh:
            break

        h = nh
    return h
```

---

## 전화번호 목록

https://programmers.co.kr/learn/courses/30/lessons/42577

### 아이디어

* 문제 그대로 구현한다.

### 코드

```python
def solution(phone_book):
    phone_book.sort()
    for ai in range(len(phone_book) - 1):
        a = phone_book[ai]
        al = len(a)
        for b in phone_book[ai + 1:]:
            if len(b) < al:
                continue

            if a == b[:al]:
                return False

    return True
```

---

## 구명보트

https://programmers.co.kr/learn/courses/30/lessons/42885

### 아이디어

* 인덱스를 2개 사용하여 몸무게가 가장 작은 사람과 가장 큰 사람 순으로 같이 태우며 보낸다.

### 코드

```python
def solution(people, limit):
    duo = 0
    people.sort()

    head = 0
    tail = len(people) - 1
    while head < tail:
        
        if people[tail] + people[head] <= limit:
            head += 1
            duo += 1
            
        tail -= 1
        
    return len(people) - duo
```

---

## 위장

https://programmers.co.kr/learn/courses/30/lessons/42578

### 아이디어

* 문제 그대로 구현한다.

### 코드

```python
from collections import defaultdict
def solution(clothes):

    dc = defaultdict(lambda: [''])

    [dc[p[1]].append(p[0]) for p in clothes]
    
    answer = 1
    for n in [len(p) for p in dc.values()]:
        answer *= n
    return answer - 1

```

---

## 카펫

https://programmers.co.kr/learn/courses/30/lessons/42842

### 아이디어

* 수학 수식을 찾아서 문제를 해결해야 한다.

### 코드

```python
def solution(brown, yellow):
    x = 3
    while True:
        y = (brown -(2 * x) + 4) / 2
        
        if (x - 2)*(y - 2) == yellow:
            return sorted([x, int(y)], reverse=True)
        
        x += 1
```

---

## 가장 큰 정사각형 찾기

https://programmers.co.kr/learn/courses/30/lessons/12905

### 아이디어

* DP 를 사용하여 해결해야 한다.
* 이 방법을 참조했다.
  * https://minnnne.tistory.com/16


### 코드

```python
def solution(board):

    answer = 0
    for i, row in enumerate(board):
        for j, c in enumerate(row):
            if c == 0 or i == 0 or j == 0:
                continue

            l = board[i][j-1]
            u = board[i-1][j]
            ul = board[i-1][j-1]

            least = min(l, u, ul)
            if least != 0:
                board[i][j] = least + 1

    maxes = [max(c) for c in [row for row in board]]
    answer = max(maxes)
    return answer**2
```


---

## 올바른 괄호

https://programmers.co.kr/learn/courses/30/lessons/12909

### 아이디어

* 여는 괄호( `(` ) 가 2번 나오면 닫는 괄호( `)` ) 가 반드시 2번 보다 적게 나와야 하는 것을 이용하여 해결하면 된다.

### 코드

```python
def solution(s):
    field = 0
    
    brackets = [1 if br == '(' else -1 for br in list(s)]
    
    if sum(brackets) != 0:
        return False
    
    for br in brackets:
        field += br
        if field < 0:
            return False
    
    return True
```


---

## 튜플

https://programmers.co.kr/learn/courses/30/lessons/64065

### 아이디어

* eval 을 사용해 실제 튜플 변수로 만들어서 해결한다.

### 코드

```python
def solution(s):
    tp = eval(f"[{s[1:-1]}]")
    tp.sort()
    
    last = tp[0]
    for i in range(1, len(tp)):
        temp = tp[i].copy()
        tp[i] -= last
        last = temp
    
    answer = [list(n)[0] for n in tp]
    
    return answer
```


---

## 다음 큰 숫자

https://programmers.co.kr/learn/courses/30/lessons/12911

### 아이디어

* 진법을 변환하는 방법을 찾아보고 해결하면 좋다.

### 코드

```python
import time
def solution(n):
    s = time.time()
    num1 = bin(n).count('1')
    while True:
        n = n + 1
        if num1 == bin(n).count('1'):
            break
    print(round(time.time() - s, 3))
    return n
```


---

## 땅따먹기

https://programmers.co.kr/learn/courses/30/lessons/12913

### 아이디어

* (i, j) 지점까지 최댓값으로 내려왔다고 했을 때 (i + 1) 라인에서 어떤 것을 밟아야 최댓값으로 갈 수 있는 지 알 수 있는데, 그 방법으로 문제를 해결한다.

### 코드

```python
def without(array, i):
    return array[:i] + array[i+1:]


def solution(land):
    field = [0]*4
    for i, row in enumerate(reversed(land)):
        temp = field.copy()
        for j, score in enumerate(row):
            field[j] = score + max(without(temp, j))
    return max(field)

```


---

## 폰켓몬

https://programmers.co.kr/learn/courses/30/lessons/1845

### 아이디어

* Counter 를 사용하여 해결한다.

### 코드

```python
from collections import Counter
def solution(nums):
    answer = 0
    counter = Counter(nums)    
    space = [list(p) for p in list(zip(counter.values(), counter.keys()))]
    space = list(reversed(space))
    
    species = set()
    for i in range(int(len(nums) / 2)):
        si = i%len(space)
        
        if space[si][0] == 0:
            continue

        space[si][0] -= 1
        species.update([str(space[si][1])])
        
    answer = len(species)
    
    return answer
```


---

## 숫자의 표현

https://programmers.co.kr/learn/courses/30/lessons/12924

### 아이디어

* 수학 수식을 찾아서 해결해야 한다.

### 코드

```python
def solution(n):
    answer = 0
    cache = 0
    for i in range(1, n):
        if (n - (cache + i)) % i == 0:
            answer += 1
        
        cache += i
        if cache >= n:
            break
    
    return answer
```


---

## 최댓값과 최솟값

https://programmers.co.kr/learn/courses/30/lessons/12939

### 아이디어

* split 을 사용하여 해결하면 된다.

### 코드

```python
def solution(s):
    
    arr = [int(n) for n in s.split(" ")]
    return f"{min(arr)} {max(arr)}"
```


---

## 최솟값 만들기

https://programmers.co.kr/learn/courses/30/lessons/12941

### 아이디어

* 문제를 천천히 읽고 해결한다.

### 코드

```python
def solution(A,B):
    A.sort()
    B.sort()
    return sum([B.pop()*n for n in A])
```


---

## 파보나치 수

https://programmers.co.kr/learn/courses/30/lessons/12945

### 아이디어

* recursive 하게 해결하면 효율성에서 문제가 생기므로 for문을 돌려서 해결한다.

### 코드

```python
def F(n):
    neck = 0
    head = 1
    for _ in range(n - 1):
        temp = head
        head += neck
        neck = temp
    return head

def solution(n):
    return F(n) % 1234567
```


---

## 행렬의 곱셈

https://programmers.co.kr/learn/courses/30/lessons/12949

### 아이디어

* 문제 그대로 읽고 해결하면 된다.

### 코드

```python
def solution(arr1, arr2):
    t = [[] for n in range(len(arr2[1]))]

    for i, row in enumerate(arr2):
        for j, p in enumerate(row):
            t[j].append(p)

    o = arr1

    answer = []

    for al in o:
        sl = []
        for bl in t:
            s = 0
            for i, a in enumerate(al):
                b = bl[i]
                s += a*b
            sl.append(s)
        answer.append(sl)

    return answer
```


---

## [카카오 인턴] 수식 최대화

https://programmers.co.kr/learn/courses/30/lessons/67257

### 아이디어

* 문제 그대로 읽고 천천히 구현한다.

### 코드

```python
from itertools import permutations
def solution(expression):
    answer = 0

    opp = list(permutations(["*", "+", "-"], 3))
    
    exp = []
    s = ""
    for ch in expression:
      if ch in ["*", "+", "-"]:
        exp.append(int(s))
        exp.append(ch)
        s = ""
      else:
        s += ch
    else:
      exp.append(int(s))
    cord = []
    for ops in opp:
      texp = exp.copy()
      for op in ops:
        for _ in range(texp.count(op)):
          opi = texp.index(op)
          
          c = texp.pop(opi + 1)
          b = texp.pop(opi)
          a = texp.pop(opi - 1)
          r = 0
          if b == "+":
            r = a + c
          elif b == "-":
            r = a - c
          elif b == "*":
            r = a * c
        
          texp.insert(opi - 1, r)

      cord.append(abs(texp[0]))

    answer = max(cord)
    return answer
```


---

## JadenCase 문자열 만들기

https://programmers.co.kr/learn/courses/30/lessons/12951

### 아이디어

* 문제 그대로 읽고 천천히 구현한다.

### 코드

```python
def solution(s):
    return " ".join([w[0].upper() + w[1:].lower() if len(w) > 0 else "" for w in s.split(" ")])
```


---

## N개의 최소공배수

https://programmers.co.kr/learn/courses/30/lessons/12953

### 아이디어

* 최소공배수를 구하는 방식을 먼저 알아야 한다.

### 코드

```python
def solution(arr):
    answer = 0
    nums = list(set(arr))
    
    m = max(nums)
    dm = m
    nums.remove(m)
    
    nums.sort(reverse=True)
    
    while True:
        for n in nums:
            if m % n != 0:
                break
        else:
            answer = m
            break

        m += dm
        
    return answer
```


---

## 짝지어 제거하기

https://programmers.co.kr/learn/courses/30/lessons/12973

### 아이디어

* stack 을 이용하여 추가하고 제거한다.

### 코드

```python
def solution(s):
    space = []
    for ch in s:
        if not len(space):
            space.append(ch)
        elif space[-1] == ch:
            space.pop()
        else:
            space.append(ch)

    return 0 if len(space) else 1

```


---

## 소수 만들기

https://programmers.co.kr/learn/courses/30/lessons/12977

### 아이디어

* combinations 을 이용하여 조합을 생성한다.
* 소수를 확인하는 공식을 알아야 한다.

### 코드

```python
from itertools import combinations
def is_prime(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
    

def solution(nums):
    combi = combinations(nums, 3)
    
    possible = [sum(p) for p in combi]
    return len([n for n in possible if is_prime(n)])
```


---

## 점프와 순간이동

https://programmers.co.kr/learn/courses/30/lessons/12980

### 아이디어

* 문제를 읽고 천천히 구현한다.

### 코드

```python
def solution(n):
    answer = 0
    k = n
    possible = [k]
    while k != 1:
        k = k // 2
        possible.append(k)

    possible.pop()
    
    teleport = lambda x: x*2
    jump = lambda x: x + 1
    
    pos = 1
    k = 1
    while True:
        if pos == k:
            if len(possible) == 0:
                break
            k = possible.pop()

        if teleport(pos) <= k:
            pos = teleport(pos)
        else:
            pos = jump(pos)
            answer += 1
    
    return answer + 1
```


---

## 영어 끝말잇기

https://programmers.co.kr/learn/courses/30/lessons/12981

### 아이디어

* 문제를 읽고 천천히 구현한다.

### 코드

```python
def solution(n, words):
    answer = [0, 0]

    last = ""
    history = {}
    
    for i, word in enumerate(words):

        if word in history or (last and word[0] != last[-1]):
            last = ""
            break
        last = word
        history[word] = True
        
    if not last:
        answer = [(i%n) + 1, i//n + 1]
    return answer
```


---

## 예상 대진표

https://programmers.co.kr/learn/courses/30/lessons/12985

### 아이디어

* 규칙을 찾아서 해결했다.

### 코드

```python
def solution(n,a,b):
    answer = 0

    for k in range(1, 21):
        size = 2**k
        start = (((a - 1) // size) * size) + 1
        end = start + size - 1
        if start <= b and b <= end:
            answer = k
            break

    return answer
```


---

## [1차] 뉴스 클러스터링

https://programmers.co.kr/learn/courses/30/lessons/17677

### 아이디어

* 합집합과 교집합
  * set_a | set_b (합집합)
  * set_a & set_b (교집합)

### 코드

```python
from collections import Counter

import re
def preprocessing(s):
    s = s.lower()
    return get_set(split_2each(s))

def split_2each(s):
    r = [s[i] + s[i+1] for i in range(0, len(s)-1) if not re.findall(r"[^\w]|[\d_]", s[i] + s[i+1])]
    return r
    
def get_set(A):
    r = set()
    for key, count in dict(Counter(A)).items():
        for i in range(count):
            r.update([key + str(i)])
    return r

def J(A, B):
    
    union = A | B
    intersection = A & B
    
    return len(intersection) / len(union) if len(union) else 1

def solution(str1, str2):
    return int(J(preprocessing(str1), preprocessing(str2)) * 65536)
```


---

## [1차] 프렌즈 4블록

https://programmers.co.kr/learn/courses/30/lessons/17679

### 아이디어

* 문제를 천천히 읽고 천천히 구현한다.

### 코드

```python
from pprint import pprint
def solution(m, n, board):
    answer = 0
    board = [list(row) for row in board]
    while True:
        removes = []
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0:
                    continue
                up = board[i - 1][j]
                left = board[i][j - 1]
                upleft = board[i - 1][j - 1]
                now = board[i][j]
                if up != "" and len(set([up, left, upleft, now])) == 1:
                    removes.extend([[i-1,j-1], [i-1,j], [i, j-1], [i, j]])

        if not removes:
            break
        for i, j in removes:
            board[i][j] = ""

        answer += len(set([str(p) for p in removes]))

        for j in range(n):
            roof = m - 1
            for i in reversed(range(m)):
                if board[i][j] != "":
                    temp = board[i][j]
                    board[i][j] = ""
                    board[roof][j] = temp

                    roof -= 1
    return answer
```


---

## [1차] 캐시

https://programmers.co.kr/learn/courses/30/lessons/17680

### 아이디어

* 문제를 천천히 읽고 천천히 구현한다.

### 코드

```python
from collections import deque
def solution(cacheSize, cities):
    answer = 0
    cache = deque([], cacheSize)
    
    for city in cities:
        city = city.lower()
        if city in cache:
            del cache[list(cache).index(city)]
            answer += 1
        else:
            answer += 5

        cache.appendleft(city)

    return answer
```


---

## 오픈채팅방

https://programmers.co.kr/learn/courses/30/lessons/42888

### 아이디어

* 문제를 천천히 읽고 천천히 구현한다.

### 코드

```python
def solution(record):
    answer = []
    cache = {}
        
    for i, p in enumerate(record):
        record[i] = p.split(" ")
    
    for command in reversed(record):
        action = command[0]
        user_id = command[1]
        
        if action != "Leave" and user_id not in cache:
            cache[user_id] = command[2]
    
    for command in record:
        nickname = cache[command[1]]
        msg = f"{nickname}님이 "
        if command[0] == "Enter":
            msg += "들어왔습니다."
        elif command[0] == "Leave":
            msg += "나갔습니다."
        else:
            continue
        
        answer.append(msg)
    
    return answer
```


---

## 후보키

https://programmers.co.kr/learn/courses/30/lessons/42890

### 아이디어

* 문제를 천천히 읽고 천천히 구현한다.

### 코드

```python
from itertools import combinations
def solution(relation):
    answer = 0
    columns = [n for n in range(len(relation[0]))]
    combos = []
    candidate_keys = []
    for n in columns:
        combos.extend(list(combinations(columns, n + 1)))
        
    for combo in combos:
        space = set()
        for candidate_key in candidate_keys:
            if not len(set(candidate_key) - set(combo)):
                break
        else:
            for row in relation:

                key = "".join([row[column] for column in combo])
                
                if key in space:

                    break

                space.update([key])
            else:
                candidate_keys.append(combo)
    answer = len(candidate_keys)
    return answer
```


---

## [3차] 방금그곡

https://programmers.co.kr/learn/courses/30/lessons/17683

### 아이디어

* 문제를 천천히 읽고 천천히 구현한다.
* `(None)` 을 반환하라고 되어있는데 글자 그대로 반환하면 된다.

### 코드

```python
def to_sec(t):
    h = t[:2]
    s = t[3:]
    
    return int(h)*60 + int(s)
    
def get_lyrics(raw):
    lyrics = []
    for ch in list(raw):
        if ch == "#":
            lyrics[-1] += ch
        else:
            lyrics.append(ch)    
        
    return lyrics

def find(lyrics, m):
    head = 0
    for code in lyrics:
        if code == m[head]:
            head += 1
        else:
            head = 0
            if code == m[head]:
                head += 1
                
        
        if head == len(m):
            return True

    return False

def solution(m, musicinfos):
    answer = ''
    candidate = {}

    for raw_info in musicinfos:
        start, end, name, raw_lyrics = raw_info.split(",")
        lyrics = get_lyrics(raw_lyrics)
        play_time = to_sec(end) - to_sec(start)
        
        while True:
            if len(lyrics) >= play_time:
                lyrics = lyrics[:play_time]
                break

            lyrics += lyrics

        if find(lyrics, get_lyrics(m)):
            if candidate:
                if candidate["play_time"] >= play_time:
                    continue

            candidate["name"] = name
            candidate["play_time"] = play_time

    answer = candidate["name"] if candidate else "(None)"
    
    return answer
```


---

## [3차] 압축

https://programmers.co.kr/learn/courses/30/lessons/17684

### 아이디어

* deque 를 이용하여 압축한다.

### 코드

```python
from collections import deque
def solution(msg):
    answer = []
    
    dic = {}
    alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i, alphabet in enumerate(alphabets):
        dic[alphabet] = i + 1
    
    deq = deque(msg)
    s = ""
    last = 0
    head = len(alphabets) + 1
    while len(deq):
        ch = deq.popleft()
        s += ch

        if s in dic:
            last = dic[s]
        else:
            answer.append(last)
            dic[s] = head
            s = ""
            last = 0
            head += 1
            deq.appendleft(ch)
    else:
        answer.append(last)
        
    return answer
```


---

## [3차] 파일명 정렬

https://programmers.co.kr/learn/courses/30/lessons/17686

### 아이디어

* 천천히 읽고 구현한다.

### 코드

```python
import re
def solution(files):
    space = []
    for order, name in enumerate(files):
        head = ""
        number =""        
        for ch in name:
            
            # ch is number?
            if ch in list("0123456789"):
                number += ch
            # ch is not number and number is exists.
            elif number:
                break
            # head
            else:
                head += ch

        space.append((head.lower(), int(number), order, name))
    
    space.sort()
    
    answer = [item[-1] for item in space]
    return answer
```


---

## [3차] n진수 게임

https://programmers.co.kr/learn/courses/30/lessons/17687

### 아이디어

* 진수 변환 방법을 알면 해결 할 수 있다.

### 코드

```python
def transform(n, number):
    space = ""
    k = number
    while k:
        q, r = divmod(k, n)
        if 10 <= r and r <= 15:
            r = list("ABCDEF")[r - 10]        
        space = f"{r}{space}"
        
        k = q
    
    return space
    
    
def solution(n, t, m, p):
    
    s = ""
    i = 1
    while len(s) < t*m:
        
        tn = transform(n, i)
        s += str(tn)
        i += 1

    s = '0' + s
    return s[p-1::m][:t]
```


---

## 추석 트래픽

https://programmers.co.kr/learn/courses/30/lessons/17676

### 코드

```python
import math
import datetime

def make_space(lines):
    space = []

    for line in lines:
        S = line[11:23]
        T = float(line[24:-1])

        # end_time = datetime.datetime.strptime("19700101 " + S, "%Y%m%d %H:%M:%S.%f").timestamp()
        end_time = datetime.datetime.strptime(S, "%H:%M:%S.%f").timestamp()
        

        start_time = end_time - T + 0.001
        space.append([start_time, end_time, 1])
        
    space = sorted(space)
    

    return space

def solution(lines):

    space = make_space(lines)
    head = 1
    space_size = len(space)
    for head in range(space_size):
        item = space[head]
        for comp_index in range(head + 1, space_size):
            comp_item = space[comp_index]

            if comp_item[0] >= item[1] + 1:
                break
                
            space[comp_index][2] += 1
    
    return max([item[2] for item in space])
```


---

## N 으로 표현

https://programmers.co.kr/learn/courses/30/lessons/42895

### 아이디어

* 8 자체로 만들 수 있는 경우를 미리 만들어 두어야 한다.

### 코드

```python
def get_needs(n):
    needs = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i + j == n:
                needs.append((i, j))

    return needs

def solution(N, number):
    answer = -1

    n_boxes = [[str(N)] * (i + 1) for i in range(8)]
    calculated = []

    for index, n_box in enumerate(n_boxes):
        n = int("".join(n_box))
        cal_item = [n]
        
        for a_index, b_index in get_needs(index + 1):
            for a_caled in calculated[a_index - 1]:
                for b_caled in calculated[b_index - 1]:
                    cal_item.append(a_caled + b_caled)
                    cal_item.append(a_caled - b_caled)
                    cal_item.append(a_caled * b_caled)
                    if b_caled:
                        cal_item.append(a_caled // b_caled)
        cal_item = list(set(cal_item))
        calculated.append(cal_item)
        if number in cal_item:
            answer = index + 1
            break

    return answer
```


---

## 2 x N 타일링

https://programmers.co.kr/learn/courses/30/lessons/12900

### 아이디어

* DP 로 해결할 수 있다.

### 코드

```python
def solution(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = (dp[i - 1] + dp[i - 2])%1000000007

    return dp[n]
```


---

## 네트워크

https://programmers.co.kr/learn/courses/30/lessons/43162

### 아이디어

* 그대로 구현하면 된다.

### 코드

```python
import numpy as np

global space
global computers
global network

def fill_space(computer_index):
    global space
    global computers
    global network
    computer = computers[computer_index]

    temp = space + computer


    indexes = np.where(np.array(temp) == 1)[0]
    for index in indexes:
        space[index] = network


    for index in indexes:
        if index == computer_index:
            continue

        fill_space(index)
    

def solution(n, _computers):
    global space
    global computers
    global network

    answer = -1
    network = -2

    space = np.array([0] * n)
    computers = _computers
    
    for head in range(len(space)):
        computer = computers[head]
        if computer[head] < 0:
            continue

        fill_space(head)

        network -= 1
    answer = len(set(space))

    return answer
```


---

## 자물쇠와 열쇠

https://programmers.co.kr/learn/courses/30/lessons/60059

### 아이디어

* 구현해야 하고 긴 문제는 클래스를 사용하면 좀 더 깔끔하게 해결 할 수 있다.

### 코드

```python
import numpy as np

class Finder():
    def __init__(self, key, lock):
        self.key = key
        self.lock = self.reverse(lock)
        self.M = len(self.key)
        self.N = len(self.lock)    
        self.spread_length = self.M * 2 + self.N - 2
        
    def reverse(self, plot):
        plot = np.array(plot)

        plot[plot == 0] = -5
        plot[plot == 1] = 2
        plot[plot == -5] = 1

        return plot


    def get_vectors(self, plot):
        vectors = []
        block_vectors = []
        for i, row in enumerate(plot):
            for j, v in enumerate(row):
                if v == 1:
                    vectors.append((i, j))
                elif v == 2:
                    block_vectors.append((i, j))

        return (vectors, block_vectors)
    
    def get_rotations(self, plot):
        plot = plot.copy()
        rotations = [plot]

        for i in range(3):
            plot = list(zip(*plot[::-1]))
            rotations.append(plot)

        return rotations
    
    def get_spread_lock(self):
        spread_lock = [[0] * self.spread_length for _ in range(self.spread_length)]
        for i, row in enumerate(self.lock):
            for j, v in enumerate(row):
                spread_lock[i + self.M - 1][j + self.M - 1] = v
        
        return spread_lock
    
    def is_piece_match(self, lvs, kvs, i, j):
        lv, blv = lvs
        kv, _ = kvs

        for li, lj in lv:
            for ki, kj in kv:
                if li == i + ki and lj == j + kj:
                    break
            else:
                return False

        for li, lj in blv:
            for ki, kj in kv:
                if li == i + ki and lj == j + kj:
                    return False

        return True


    def is_match(self, lvs, kvs):

        for i in range(self.spread_length - self.M + 1):
            for j in range(self.spread_length - self.M + 1):
                if self.is_piece_match(lvs, kvs, i, j):
                    return True
                    
        return False

    def find(self):

        spread_lock = self.get_spread_lock()
        lvs = self.get_vectors(spread_lock)
        if len(lvs[0]) == 0:
            return True

        for rotation in self.get_rotations(self.key):
            kvs = self.get_vectors(rotation)
            if self.is_match(lvs, kvs):
                return True

        return False        

def solution(key, lock):
    return Finder(key, lock).find()
```


---

## 정수 삼각형

https://programmers.co.kr/learn/courses/30/lessons/43105

### 아이디어

* DP 를 사용하여 해결 할 수 있다.
* Level 2 의 문제인 `땅따먹기`와 비슷하다.

### 코드

```python
def solution(triangle):
    
    for i in reversed(range(len(triangle) - 1)):
        row = triangle[i]
        next_row = triangle[i + 1]
        for j, _ in enumerate(row):
            row[j] += max(next_row[j], next_row[j + 1])
    
    return triangle[0][0]
```


---

## 디스크 컨트롤러

https://programmers.co.kr/learn/courses/30/lessons/42627

### 아이디어

* heap 을 사용하여 해결 할 수 있다.

### 코드

```python
import heapq

def solution(jobs):
    
    head = 0
    heap = []
    size = len(jobs)
    future = 0
    answer = 0

    jobs = sorted(jobs)
    
    while True:
        
        if head < size and jobs[head][0] > future and len(heap) == 0:
            future = jobs[head][0]

        for i in range(head, size):
            if future < jobs[head][0]:
                break

            heapq.heappush(heap, (jobs[i][1], jobs[i][0]))
            head += 1

        period, start_time = heapq.heappop(heap)
        latency_time = future - start_time
        answer += latency_time + period

        future += period

        if len(heap) == 0 and head >= size:
            break

    return answer // size
```


---

## 단속카메라

https://programmers.co.kr/learn/courses/30/lessons/42884

### 아이디어

* 그리디하게 해결 할 수 있다.
* 규칙을 찾아야 한다.

### 코드

```python
def solution(routes):
    answer = 0
    last = -1e9

    routes = sorted(routes)

    for start, end in routes:
        if start > last:
            last = end
            answer += 1
        elif end < last:
            last = end
    
    return answer
```


---

## 섬 연결하기

https://programmers.co.kr/learn/courses/30/lessons/42861

### 아이디어

* 서로소와 크루스칼로 해결 할 수 있다.

### 코드

```python
import heapq
def find_parent(parent, x):
    if parent[x] != x:
        return find_parent(parent, parent[x])

    return parent[x]

def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    
    if a < b:
         parent[b] = a
    else:
        parent[a] = b
    

def solution(n, costs):

    parent = list(range(n))
    answer = 0

    heap = []
    for a, b, cost in costs:
        heapq.heappush(heap, (cost, a, b))
        

    while len(heap):
        cost, a, b = heapq.heappop(heap)

        if find_parent(parent, a) == find_parent(parent, b):
            continue

        union_parent(parent, a, b)
        
        answer += cost

    return answer
```


---

## 가장 먼 노드

https://programmers.co.kr/learn/courses/30/lessons/49189

### 아이디어

* 플로이드 워셜로는 효율성을 통과 할 수 없다.
* BFS를 사용하면 해결 할 수 있다.

### 코드

```python
from collections import defaultdict
def solution(n, edge):
    answer = 0

    graph = defaultdict(set)

    for a, b in edge:
        graph[a].add(b)
        graph[b].add(a)

    box = {1}
    history = {1}
    while len(box):
        answer = len(box)

        tmp = set()
        for id in box:
            history.add(id)
            tmp = tmp | graph[id]

        box = {id for id in tmp if id not in history}
        
    return answer
```


---

## 단어 변환

https://programmers.co.kr/learn/courses/30/lessons/43163

### 아이디어

* 다익스트라 알고리즘으로 해결 할 수 있다.

### 코드

```python
from collections import Counter
import heapq
import numpy as np
def dijkstra(begin, target, relations):
    INF = int(1e10)
    heap = []
    heapq.heappush(heap, (0, begin))

    distances = {key:INF for key in relations.keys()}
    distances[begin] = 0
    while len(heap):
        distance, key = heapq.heappop(heap)
        if distances[key] < distance:
            continue

        relation = relations[key]
        for rel_key in relation:
            rel_distance = relation[rel_key]
            sum_distance = distance + rel_distance

            if sum_distance < distances[rel_key]:
                distances[rel_key] = sum_distance
                heapq.heappush(heap, (sum_distance, rel_key))
        if key == target:
            break

    
    return distances[target] if target in distances else 0

    
def has_relation(a, b):
    count = 0
    for ai, ap in enumerate(a):
        bp = b[ai]
        if ap == bp:
            count += 1

    return len(a) - count == 1

def get_relations(words):
    relations = {}
        
    for a in words:
        relations[a] = {}
        for b in words:
            if has_relation(a, b):
                relations[a][b] = 1

    return relations

def solution(begin, target, words):
    answer = 0
    words.append(begin)
    relations = get_relations(words)
    
    answer = dijkstra(begin, target, relations)
    
    
    return answer
```

---

## 이중우선순위큐

https://programmers.co.kr/learn/courses/30/lessons/42628

### 아이디어

* min heap 과 max heap 을 사용하여 해결 할 수 있다.

### 코드

```python
import heapq

def pop(maxheap, minheap, history, op_b):
    v = None
    while True:
        if op_b == "1":
            if len(maxheap) == 0:
                break
            item = heapq.heappop(maxheap)
            item[0] = -item[0]
            
        else:
            if len(minheap) == 0:
                break                    
            item = heapq.heappop(minheap)

        if item[1] not in history:
            history.add(item[1])
            v = item[0]
            break
    
    return v

def solution(operations):
    answer = [0, 0]
    minheap = []
    maxheap = []
    history = set()
    
    for index, operation in enumerate(operations):
        op_a, op_b = operation.split(" ")
        if op_a == "I":
            op_b = int(op_b)
            heapq.heappush(minheap, [op_b, index])
            heapq.heappush(maxheap, [-op_b, index])
            
        elif op_a == "D":
            
            pop(maxheap, minheap, history, op_b)

                
    v = pop(maxheap, minheap, history, "1")
    answer[0] = v if v else 0
    v = pop(maxheap, minheap, history, "-1")
    answer[1] = v if v else 0
        
    return answer
```


---

## 입국심사

https://programmers.co.kr/learn/courses/30/lessons/43238

### 아이디어

* 이진탐색으로 해결 할 수 있다.

### 코드

```python
def solution(n, times):
    answer = 0
    
    l = 1
    r = max(times) * n

    while l <= r:
        target = (l + r) // 2

        people = sum([target // t for t in times])

        if people < n:
            l = target + 1
        else:
            r = target - 1
            answer = target

    return answer

solution(6, [7, 10])
```


---

## 여행경로

https://programmers.co.kr/learn/courses/30/lessons/43164

### 아이디어

* 한 번 가서 돌아올 수 없는 경우는 마지막으로 갈 때밖에 없다.

### 코드

```python
from collections import defaultdict
def solution(tickets):
    routes = defaultdict(list)
    stack = ['ICN']
    path = []

    for a, b in tickets:
        routes[a].append(b)
    
    for key, route in routes.items():
        routes[key] = sorted(route, reverse=True)

    while len(stack):
        head = stack[-1]
        
        if head not in routes or len(routes[head]) == 0:
            path.append(stack.pop())
        else:
            stack.append(routes[head].pop())
    
    return list(reversed(path))
```


---

## 등굣길

https://programmers.co.kr/learn/courses/30/lessons/42898

### 아이디어

* 갈 수 있는 방법의 경우의 수를 모든 칸에 적용한다.
* DP 를 사용하여 해결 할 수 있다.

### 코드

```python
def solution(m, n, puddles):
    head = (0, 0)
    puddles = [[p[1], p[0]] for p in puddles]

    matrix = {}
    
    for i in range(n):
        for j in range(m):
            if [i + 1, j + 1] not in puddles: 
                matrix[(i, j)] = 0
    matrix[(0, 0)] = 1
    
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            if (i, j) in matrix:
                matrix[(i, j)] = (matrix.get((i - 1, j), 0) + matrix.get((i, j - 1), 0)) % 1000000007

    return matrix[(n - 1, m - 1)]
```


---

## 베스트앨범

https://programmers.co.kr/learn/courses/30/lessons/42579

### 아이디어

* 문제를 천천히 구현하면 된다.

### 코드

```python
from collections import defaultdict
def solution(genres, plays):

    answer = []
    
    space = defaultdict(list)
    # dict 장르를 키로 넣는다.
    # 값은 (플레이수, 고유번호) 로 넣는다.
    for uid, (genre, play) in enumerate(zip(genres, plays)):
        space[genre].append((play, -uid))
    
    # dict 에 장르 별 총합을 구해서 정렬하고 for문을 돌린다.
    genres_space = [(sum([info[0] for info in infos]), genre) for genre, infos in space.items()]
        
    # 그 장르의 데이터를 정렬한다.
    # for 문으로 고유번호를 순서대로 결과값에 넣는다.
    for _, genre in sorted(genres_space, reverse=True):
        items = []
        for index, item in enumerate(sorted(space[genre], reverse=True)):
            if index == 2: break
            items.append(-item[1])

        answer.extend(items)
        
    return answer
```


---

## 순위



### 아이디어

* loser 가 이긴 애들은 winner 도 이긴다.
* winer 가 진 애들은 loser 도 진다.

### 코드

```python
import numpy as np
def solution(n, results):
    answer = 0

    relations = [np.array([0] * n) for i in range(n)]

    results = [[a-1, b-1] for a, b in results]

    for a, b in results:
        relations[a][b] = 1
        relations[b][a] = -1

    for relation in relations:
        
        wins = [p[0] for p in np.argwhere(relation == 1)]
        loses = [p[0] for p in np.argwhere(relation == -1)]


        for win_index in wins:
            loser = relations[win_index]
            loser[relation == -1] = -1

        for lose_index in loses:
            winner = relations[lose_index]
            winner[relation == 1] = 1
            
    answer = len(list(filter(lambda x: np.count_nonzero(x == 0) == 1, relations)))
    return answer
```


---

## 기둥과 보 설치

https://programmers.co.kr/learn/courses/30/lessons/60061

### 아이디어

* 천천히 구현한다.
* 삭제 구현시만 조심하면 되는데, 삭제 후에도 멀쩡한지 전부 확인하여 체크해보아야 한다.

### 코드

```python
def install_check(kind, x, y, pillars, beams):
    if kind == 0:
        if y == 0:
            return True
        if (x, y - 1) in pillars:
            return True
        if (x - 1, y) in beams or (x, y) in beams:
            return True

    elif kind == 1:
        if (x - 1, y) in beams and (x + 1, y) in beams:
            return True
        if (x, y - 1) in pillars or (x + 1, y - 1) in pillars:
            return True

    return False

def solution(n, build_frame):
    answer = [[]]
    pillars = set()
    beams = set()

    # 0 기둥
    # 1 보

    for x, y, a, b in build_frame:

        # 설치
        if b == 1:
            if not install_check(a, x, y, pillars, beams):
                continue

            if a == 0:
                pillars.add((x, y))
            else:
                beams.add((x, y))

        # 삭제
        else:

            if a == 0:

                pillars.remove((x, y))

                for px, py in pillars:
                    if not install_check(0, px, py, pillars, beams):
                        pillars.add((x, y))
                        break

                for bx, by in beams:
                    if not install_check(1, bx, by, pillars, beams):
                        pillars.add((x, y))
                        break


            else:
                beams.remove((x, y))

                for px, py in pillars:
                    if not install_check(0, px, py, pillars, beams):
                        beams.add((x, y))
                        break

                for bx, by in beams:
                    if not install_check(1, bx, by, pillars, beams):
                        beams.add((x, y))
                        break

    
    answer = sorted([[x, y, 1] for x, y in beams] + [[x, y, 0] for x, y in pillars])
    
    return answer
```


---

## 가장 긴 팰린드롬

https://programmers.co.kr/learn/courses/30/lessons/12904

### 아이디어

* 모두 확인한다.

### 코드

```python
def is_palidrome(s):
    length = len(s)
    for index in range(length // 2):
        if s[index] != s[length - index - 1]:
            return False
    return True


def solution(s):
    answer = 0
    last = ""
    
    for i in range(len(s)):
        candidate = s[i:i + len(last)]
            
        for j in range(i + len(last), len(s)):
            candidate += s[j]
            if is_palidrome(candidate):
                last = candidate
            
    return len(last)
```


---

## 외벽 점검

https://programmers.co.kr/learn/courses/30/lessons/60062

### 아이디어

* 가장 많이 처리 할 수 있는 친구부터 모든 약한 부위를 돌며 처리 할 수 있는 모든 경우를 거미줄처럼 확장시켜나간다.

### 코드

```python
def solution(n, weak, dist):
    answer = 0

    dist = reversed(dist)

    availables = [()]
    
    for d in dist:
        answer += 1

        groups = []
        for i, w in enumerate(weak):
            cand = weak[i:] + [v + n for v in weak[:i]]
            groups.append(set([c % n for c in cand if c <= d + w]))

        next = set()

        for group in groups:
            for av in availables:
                new = group | set(av)
                if len(new) == len(weak):
                    return answer
                next.add(tuple(new))
                
        availables = next

    return -1
```


---

## 블록 이동하기

https://programmers.co.kr/learn/courses/30/lessons/60063

### 아이디어

* 문제를 그대로 구현하지 않으면 문제가 생긴다.
* 문제를 천천히 그대로 구현한다.

### 코드

```python
from collections import deque
def get_can_moves(a, b, board):
    moves = []


    # 회전
    # 가로 방향 일 때
    if a[0] == b[0]:
        for d in [1, -1]:
            if board[a[0] + d][a[1]] == 0 and board[b[0] + d][b[1]] == 0:
                moves.append((a, (a[0] + d, a[1])))
                moves.append((b, (b[0] + d, b[1])))

    # 세로 방향 일 때
    else:
        for d in [1, -1]:
            if board[a[0]][a[1] + d] == 0 and board[b[0]][b[1] + d] == 0:
                moves.append((a, (a[0], a[1] + d)))
                moves.append((b, (b[0], b[1] + d)))
    # 평행이동
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if board[a[0] + di][a[1] + dj] == 0 and board[b[0] + di][b[1] + dj] == 0:
            moves.append(((a[0] + di, a[1] + dj), (b[0] + di, b[1] + dj)))

    return moves    

    
def get_new_board(board):
    temp = [[1] * (len(board) + 2) for _ in range(len(board) + 2)]
    for i in range(len(board)):
        for j in range(len(board)):
            temp[i + 1][j + 1] = board[i][j]
    return temp

def solution(board):


    deq = deque()
    history = set()
    
    new_board = get_new_board(board)
    deq.append([(1, 1), (1, 2), 0])

    n = len(board)

    while len(deq):
        a, b, count = deq.popleft()
        if a == (n, n) or b == (n, n):
            return count

        for nxt in get_can_moves(a, b, new_board):
            if nxt not in history:
                deq.append([*nxt, count + 1])
                history.add(nxt)

    return False
```


---

## 거스름돈

https://programmers.co.kr/learn/courses/30/lessons/12907

### 아이디어

* DP(table) 를 이용하여 해결 할 수 있다.

### 코드

```python
from collections import defaultdict
def solution(n, money):

    table = [[1] + [0] * (n) for _ in range(len(money))]

    money = sorted(money)
    for mi, m in enumerate(money):
        for target in range(1, len(table[mi])):
            v = 0
            v += table[mi - 1][target] if m >= 1 else 0

            if target >= m:
                v += table[mi][target - m]
            else:
                v += 1 if target % m == 0 else 0

            table[mi][target] = v % 1000000007

    return table[mi][n]
```


---

## 멀리 뛰기

https://programmers.co.kr/learn/courses/30/lessons/12914

### 코드

```python
import math
def solution(n):
    answer = 1
    f = math.factorial
    
    for i in range(1, (n // 2) + 1):
        answer += f(n - i) // int(f(n - (2 * i)) * f(i))
        
    return answer % 1234567
```


---

## 방문 길이

https://programmers.co.kr/learn/courses/30/lessons/49994

### 아이디어

* 그대로 구현하면 된다.

### 코드

```python
def solution(dirs):
    answer = 0
    history = set()
    x = 0
    y = 0

    for ch in dirs:
        nx = x
        ny = y
        if ch == "U":
            ny += 1
            
        elif ch == "L":
            nx -= 1

        elif ch == "R":
            nx += 1

        elif ch == "D":
            ny -= 1
        

        if nx in [6, -6] or ny in [6, -6]:
            continue

        cord = (nx + x, ny + y)

        if cord not in history:
            answer += 1
            history.add(cord)

        x = nx
        y = ny


    return answer
```


---

## 불량 사용자

https://programmers.co.kr/learn/courses/30/lessons/64064

### 아이디어

* 여덟 퀸 문제의 해법으로 해결 할 수 있다.

### 코드

```python
histories = set()

def check(history, groups, index, n):

    global histories

    if n == index:
        histories.add(tuple(sorted(list(history))))
        return

    for name in groups[index]:
        if name in history:
            continue

        history.add(name)
        check(history, groups, index + 1, n)
        history.remove(name)


def solution(user_id, banned_id):
    answer = 0
    
    banned_indices = []
    
    for b in banned_id:
        
        indices = [(i, ch) for i, ch in enumerate(b) if ch != "*"]
        banned_indices.append((indices, len(b)))
        
    groups = []
    for indices, length in banned_indices:
        group = set()
        for u in user_id:
            if length != len(u):
                continue

            for i, ch in indices:
                if u[i] != ch:
                    break
            else:
                group.add(u)
        groups.append(group)

    history = set()
    check(history, groups, 0, len(groups))
    answer = len(histories)
    return answer
```


---

## 야근 지수

https://programmers.co.kr/learn/courses/30/lessons/12927

### 아이디어

* heap으로 해결 할 수 있다.

### 코드

```python
import heapq
def solution(n, works):
    answer = 0
    heap = [-w for w in works]
    heapq.heapify(heap)

    for _ in range(n):
        v = heapq.heappop(heap)
        v += 1

        if v != 0:
            heapq.heappush(heap, v)

        if len(heap) == 0:
            break
    answer = sum([w**2 for w in heap])

    return answer
```


---

## 줄 서는 방법

https://programmers.co.kr/learn/courses/30/lessons/12936

### 아이디어

* 하나씩 직접 하면 효율성 문제가 발생한다.
* 규칙을 찾아서 해결해야 한다.

### 코드

```python
def solution(n, k):
    k -= 1
    answer = []

    cases_map = {1: 1}
    for i in range(2, n + 1):
        cases_map[i] = cases_map[i - 1] * i

    size = n
    
    number_map = list(range(1, n + 1))
    for i in range(size - 1):
        
        nth = (k // cases_map[n - 1])
        number = number_map.pop(nth)
        answer.append(number)

        k -= nth * cases_map[n - 1]
        
        n -= 1

    answer.append(number_map[k])

    return answer
```


---

## 최고의 집합

https://programmers.co.kr/learn/courses/30/lessons/12938

### 코드

```python
def solution(n, s):
    answer = []
    arr = [s // n] * n
    remain = s - sum(arr)

    for i in range(remain):
        arr[i % n] += 1
    
    if all(arr):
        answer = arr
    else:
        answer = [-1]

    answer = sorted(answer)

    return answer
```


---

## 하노이의 탑

https://programmers.co.kr/learn/courses/30/lessons/12946

### 아이디어

* 직접 아이디어를 떠올릴 수는 없을 것 같다.
* 방식을 찾아보고 해결하는 편이 낫다.

### 코드

```python
def hanoi(n, start, via, to):
    move = []
    if n == 1:
        return [[start, to]]
    
    move += hanoi(n - 1, start, to, via)
    move += [[start, to]]
    move += hanoi(n - 1, via, start, to)

    return move


def solution(n):
    return hanoi(n, 1, 2, 3)
```


---

## 징검다리 건너기

https://programmers.co.kr/learn/courses/30/lessons/64062

### 아이디어

* 문제 그대로 천천히 구현하면 된다.

### 코드

```python
import heapq
def solution(stones, k):
    answer = 0
    
    groups = {}
    
    heap = [(stone, index) for index, stone in enumerate(stones)]

    heapq.heapify(heap)

    while len(heap):
        stone, index = heapq.heappop(heap)
        
        front = None
        back = None
        # 첫번째 인덱스가 아니면
        if index != 0 and index - 1 in groups:
            front = groups[index - 1]
            
        # 마지막 인덱스가 아니면
        if index != len(stones) - 1 and index + 1 in groups:
            back = groups[index + 1]


        if front and back:
            fg = groups[index - 1]
            bg = groups[index + 1]

            fg[1] = bg[1]

            for i in range(bg[0] - 1, bg[1] + 1):
                groups[i] = fg

        elif not front and not back:
            groups[index] = [index, index]
        elif front:
            groups[index] = groups[index - 1]
            groups[index][1] = index
        elif back:
            groups[index] = groups[index + 1]
            groups[index][0] = index

        if k <= groups[index][1] - groups[index][0] + 1:
            break
    
    answer = max([stones[i] for i in range(groups[index][0], groups[index][1] + 1)])

    return answer
```


---

## N-Queen

https://programmers.co.kr/learn/courses/30/lessons/12952

### 아이디어

* 하나씩 직접 실행해보려고 할 때, set 을 사용하여 해결하면 쉽게 해결 할 수 있다.

### 코드

```python
def nqueen(n, j, line, diag, rdiag):
    complete = 0
    if j == n:
        return 1

    for i in range(n):
        if i in line or i + j in diag or i - j in rdiag:
            continue
        line.add(i)
        diag.add(i + j)
        rdiag.add(i - j)
        complete += nqueen(n, j + 1, line, diag, rdiag)
        line.remove(i)
        diag.remove(i + j)
        rdiag.remove(i - j)

    return complete

def solution(n):
    line = set()
    diag = set()
    rdiag = set()

    return nqueen(n, 0, line, diag, rdiag)
```


---

## 보석 쇼핑

https://programmers.co.kr/learn/courses/30/lessons/67258

### 아이디어

* 인덱스를 2개 사용하여 해결하면 된다.

### 코드

```python
from collections import deque

def solution(gems):
    answer = [-1, -1]

    kinds = {gem: 0 for gem in list(set(gems))}
    
    required = {gem for gem in gems}
    left = 0
    sufficients = []
    for right, gem in enumerate(gems):

        kinds[gem] += 1

        if gem in required:
            required.remove(gem)

        if len(required) == 0:
            for hindex in range(left, right + 1):
                hgem = gems[hindex]
                if kinds[hgem] > 1:
                    kinds[hgem] -= 1
                    left += 1
                else:
                    sufficients.append((left, right))
                    break

    min_value = 1e9
    for a, b in sufficients:
        if b - a < min_value:
            min_value = b - a
            answer = [a + 1, b + 1]
        
    return answer
```


---

## 배달

https://programmers.co.kr/learn/courses/30/lessons/12978

### 아이디어

* 다익스트라를 사용하면 쉽게 해결 할 수 있다.

### 코드

```python
from collections import defaultdict
import heapq

def dijkstra(N, graph):
    INF = int(1e9)
    table = [0] + ([INF] * (N - 1))
    heap = []
    
    # distance, index
    now = 0
    heapq.heappush(heap, (0, 0))

    while len(heap):
        distance, index = heapq.heappop(heap)

        if distance > table[index]:
            continue
        
        for node_index, node_distance in graph[index].items():
            new_distance = node_distance + table[index]
            if new_distance < table[node_index]:
                heapq.heappush(heap, (new_distance, node_index))
                table[node_index] = new_distance
    return table

def solution(N, road, K):
    answer = 0
    
    graph = defaultdict(dict)

    for start, end, price in road:
        a = start - 1
        b = end - 1
        if a in graph and b in graph[a]:
            if price > graph[a][b]:
                continue

        graph[a][b] = price
        graph[b][a] = price

    table = dijkstra(N, graph)
    answer = len(list(filter(lambda x: x <= K, table)))

    return answer
```


---

## 경주로 건설

https://programmers.co.kr/learn/courses/30/lessons/67259

### 아이디어

* BFS 로 해결해야 한다.
* 이미 존재하는지 확인할 때 가격이 더 낮으면 그대로 진행하는 방향으로 해야 제대로된 가격이 나온다.

### 코드

```python
from collections import deque

def visitable(N, i, j, board):
    return 0 <= i < N and 0 <= j < N and board[i][j] == 0
    

def solution(board):
    
    d = 3
    q = deque()
    N = len(board)
    I, J, DIRECTION, SUM = 0, 1, 2, 3
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    deltas = [(-1, 0, UP), (1, 0, DOWN), (0, -1, LEFT), (0, 1, RIGHT)]
    confirm = {}
    confirm[(0, 0)] = 0

    if board[0][1] == 0:
        confirm[(0, 1)] = 100
        q.append([0, 1, RIGHT, 100])

    if board[1][0] == 0:
        confirm[(1, 0)] = 100
        q.append([1, 0, DOWN, 100])
    
    while len(q):
        cur = q.popleft()

        for di, dj, dd in deltas:
            new = [cur[I] + di, cur[J] + dj, dd, cur[SUM]]

            if new[DIRECTION] == cur[DIRECTION]:
                new[SUM] += 100
            else:
                new[SUM] += 600

            if visitable(N, new[I], new[J], board):

                if (new[I], new[J]) in confirm and confirm[(new[I], new[J])] < new[SUM]:
                    continue

                confirm[(new[I], new[J])] = new[SUM]
                q.append(new)

    return confirm[(N-1, N-1)]
```


---

## 기지국 설치

https://programmers.co.kr/learn/courses/30/lessons/12979

### 아이디어

* ceil 을 사용하여 해결 할 수 있다.

### 코드

```python
import math
import numpy as np

def solution(n, stations, w):
    answer = 0
    size = len(stations)
    space = [[0, -1]]

    for i, s in enumerate(stations):
        space[i][1] = s - w - 2
        space.append([s + w, -1])


    space[-1][1] = n - 1
    wide = w * 2 + 1
    for start, end in space:
        if start > end:
            continue
        
        v = (end - start + 1) / wide
        answer += math.ceil(v)

    return answer
```


---

## 숫자 게임

https://programmers.co.kr/learn/courses/30/lessons/12987

### 아이디어

* heap 을 두개 사용하여 해결 할 수 있다.

### 코드

```python
import heapq

def solution(A, B):
    answer = 0
    ha = [-n for n in A.copy()]
    hb = [-n for n in B.copy()]

    heapq.heapify(ha)
    heapq.heapify(hb)

    while len(ha):
        b = -heapq.heappop(hb)
        while len(ha):
            a = -heapq.heappop(ha)

            if b > a:
                answer += 1
                break
    
    return answer
```


---

## [1차] 셔틀버스

https://programmers.co.kr/learn/courses/30/lessons/17678

### 아이디어

* 그대로 구현하면 해결 할 수 있다.

### 코드

```python
import heapq
def to_minute(s):
    h = int(s[:2])
    m = int(s[3:])

    return (h * 60) + m

def to_formattime(minute):
    return str(minute // 60).zfill(2) + ":" + str(minute % 60).zfill(2)

    

def solution(n, t, m, timetable):
    answer = ''
    crews = [to_minute(s) for s in timetable]
    heapq.heapify(crews)
    shuttles = [[540, []]]

    for i in range(1, n):
        shuttles.append([shuttles[i - 1][0] + t, []])

    shuttle_index = 0
    while len(crews):
        crew = heapq.heappop(crews)

        for si in range(shuttle_index, len(shuttles)):
            arrive, rides = shuttles[si]

            if arrive >= crew:
                rides.append(crew)
                shuttle_index = si
                if len(rides) == m:
                    shuttle_index += 1                
                break

    arrive, candidates = shuttles[-1]

    if len(candidates) < m:
        answer = arrive
    else:
        answer = candidates[-1] - 1

    answer = to_formattime(answer)
    return answer
```


---

## 길 찾기 게임

https://programmers.co.kr/learn/courses/30/lessons/42892

### 아이디어

* `sys.setrecursionlimit(10**6)` 를 해주어야 오류가 발생하지 않는다.
* 트리를 구현 할 수 있어야 한다.

### 코드

```python
import sys
sys.setrecursionlimit(10**6)
class Tree():
    def __init__(self, nodeinfo):
        self.data = max(nodeinfo, key=lambda x: x[1])
        
        left_nodeinfo = list(filter(lambda x: x[0] < self.data[0], nodeinfo))
        right_nodeinfo = list(filter(lambda x: x[0] > self.data[0], nodeinfo))

        self.left = Tree(left_nodeinfo) if left_nodeinfo else None
        self.right = Tree(right_nodeinfo) if right_nodeinfo else None

    def preorder_traversal(self, node, path):
        path.append(node.data)
        if node.left: 
            self.preorder_traversal(node.left, path)
        if node.right: 
            self.preorder_traversal(node.right, path)

    def postorder_traversal(self, node, path):
        if node.left: 
            self.postorder_traversal(node.left, path)
        if node.right: 
            self.postorder_traversal(node.right, path)

        path.append(node.data)

        
def solution(nodeinfo):
    tree = Tree(nodeinfo)
    prepath = []
    postpath = []
    tree.preorder_traversal(tree, prepath)
    tree.postorder_traversal(tree, postpath)    
    dic = {tuple(v): i + 1 for i, v in enumerate(nodeinfo)}
    
    return [[dic[tuple(v)] for v in prepath], [dic[tuple(v)] for v in postpath]]
```


---

## 매칭 점수

https://programmers.co.kr/learn/courses/30/lessons/42893

### 아이디어

* 문제를 잘 읽으면 쉽게 해결 할 수 있다.

### 코드

```python
from collections import defaultdict
import re

def get_base_score(s, word):
    score = 0
    for m in re.finditer(word, s, re.I):
        span = m.span()
        front = span[0] - 1
        back = span[1]

        if front >= 0 and s[front].isalpha():
            continue

        if back < len(s) and s[back].isalpha():
            continue
        
        score+=1

    return score

def get_property(s, p):
    s = s[s.find(p):]
    s = s[s.find('"') + 1:]
    s = s[:s.find('"')]
    return s    

def solution(word, pages):

    BASE_SCORE, EXTERNAL_LINK, LINK_SCORE, MATCHING_SCORE, INCOMING_LINKS = list(range(0, 5))

    items = defaultdict(lambda: [0, 0, 0, 0, []])
    
    urls = []
    for page in pages:
        
        url = get_property(re.search(r"<meta property=\"og:url\" content=\".*?\"/>", page).group(), "content")

        item = items[url]
        item[BASE_SCORE] = get_base_score(page, word)

        for a_content in re.findall(r"<a .*?>", page):
            external_url = get_property(a_content, "href")
            item[EXTERNAL_LINK] += 1
            items[external_url][INCOMING_LINKS].append(url)
    
        urls.append(url)
        
    for url in urls:
        item = items[url]
        for incoming_url in item[INCOMING_LINKS]:
            if items[incoming_url][EXTERNAL_LINK] != 0:
                item[LINK_SCORE] += items[incoming_url][BASE_SCORE] / items[incoming_url][EXTERNAL_LINK]
        item[MATCHING_SCORE] = item[LINK_SCORE] + item[BASE_SCORE]

    max_score = -1
    for index, url in enumerate(urls):
        matching_score = items[url][MATCHING_SCORE]
        if matching_score > max_score:
            max_score = matching_score
            answer = index

    return answer
```


---

## 풍선 터뜨리기

https://programmers.co.kr/learn/courses/30/lessons/68646

### 아이디어

* 왼쪽에 1개 이상 오른쪽에 1개 이상 어떤 수보다 작은 것이 있으면 그 숫자는 터질 수 없는 풍선이다.
* 가장 작은 수를 계속 업데이트 해나가며 왼쪽에 아무것도 없을 경우 정답 + 1을 한다.
* 오른쪽은 지속적으로 확인해나가야 하므로 heapq 를 사용하여 가장 큰 수보다 작은 수가 나오면 그 수를 제거한다.
* 마지막까지 heap 에 살아남은 풍선들은 살아남을 수 있는 풍선들이다.

### 코드

```python
import heapq
def solution(a):
    answer = 0
    left_min = 1e9
    heap = []
    
    for n in a:
        while len(heap):
            cand = -heapq.heappop(heap)
            if cand < n:
                heapq.heappush(heap, -cand)
                break

        if left_min > n:
            left_min = n
            answer += 1
        else:
            heapq.heappush(heap, -n)

    answer += len(heap)
    
    return answer
```

---
