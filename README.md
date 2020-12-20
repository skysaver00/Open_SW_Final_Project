# Final_Project_OPEN_SW
- 소프트웨어학과 201620931 김영우

# 최종 프로젝트 강의 내용
- 파이썬 패키지 Numpy, Pandas설명 그리고 함수사용법과 기초적인 수치해석

# 필요한 준비물
- 기본적인 컴퓨터 지식
- 윈도우, 리눅스 운영체제가 설치된 컴퓨터 (강의는 윈도우를 이용할 예정)
- 파이썬 3.9를 제외한 버전 설치 (Python 3.8, 3.7을 추천. 강의는 Python 3.8.6을 이용할 예정)

**파이썬 3.9를 추천하지 않는 이유는 무엇인가요?**

**파이썬 3.9는 2020년 10월에 처음 출시된, 지금까지 나온 파이썬 버전들 중에서 가장 최신 버전입니다.**
**현재 파이썬 3.9는 안정성이 이전에 출시된 버전들에 비해 부족하고, Numpy, Pandas를 PIP등으로 설치하는 과정에서 오류가 보고되고 있습니다. 이런이유로 파이썬 3.9는 추천하지 않습니다.**

# Numpy란 무엇인가?
Numpy 또는 넘파이는 파이썬 프로그래밍 언어를 위한 라이브러리입니다. 

Numpy는 파이썬에 1차원 이상의 배열, 행렬을 할 수 있게 지원해주고, 이러한 배열 등에서 작동하는 수학, 과학 연산도 추가해줍니다.

파이썬의 다른 패키지인 SciPy, Matplotlib, Pandas와 같이, 이들의 패키지를 같이 조합해서 사용하면, MATLAB과 같은 전문적인 수치해석 공학 소프트웨어처럼 사용 가능합니다.

**Numpy는 오픈소스 소프트웨어입니다. 자유롭게 쓸 수 있습니다.** 이것에 대해 가장 큰 장점은 수치해석 프로그램인 MATLAB과 비교해서 비용이 거의 들지 않는다는 장점에, MATLAB에서 제공하는 함수들을 거의 모두 사용할 수 있다는 것입니다.

# Numpy를 사용하는 이유
- 빠르다.
- 앞서 말한 Pandas, Matplotlib, SciPy와 같이 써서 더 큰 시너지를 만들 수 있다.
- 배열, 행렬을 지원해서 행렬 산술이 가능해서, 매우 편리하게 사용 가능하다.
- Numpy에서 지원하는 함수들이 많아서 다양한 기능을 쓸 수 있다.

# Numpy VS 파이썬 List
파이썬의 List는 파이썬이 기본적으로 제공하는 Numpy의 배열과 비교가 되는 데이터 구조입니다.

Numpy가 파이썬의 List보다 훨씬 빠릅니다. 파이썬의 List는 포인터 주소값이 들어가있어서 리스트의 주소를 넣어두고, 그 주소로 메모리를 찾아가는 반면에,

Numpy의 배열은 C언어의 배열처럼 데이터에 값이 직접들아갑니다. 이를 통해 고정된 값을 가지며, 파이썬의 List보다 더 빠른 연산이 가능합니다.

또한 Numpy는 List보다 지원하는 함수, 기능들이 더 많아 사용하기 편합니다.
같은 값을 저장할때, Numpy가 더 적은 용량으로 저장할 수 있습니다.

따라서 Numpy는 기본 파이썬 List에 비해, 성능, 사이즈, 편리성에서 더 유리합니다.

# Pandas란 무엇인가?
Pandas 또는 판다스는 파이썬에서 사용하는 데이터 분석 라이브러리입니다

Pandas는 테이블 형식의 데이터를 쉽게 처리할 수 있게 해줍니다.

**Pandas역시 오픈소스 소프트웨어입니다.**

# Pandas의 자료구조
Pandas에서 사용하는 자료구조는 크게 두가지로 나뉩니다 첫번째는 Series, 두번째는 Data Frame입니다.

두 자료구조의 가장 큰 차이점은 Series는 1차원 배열과 같고, Data Frame은 2차원 배열과 같은 형태를 가진다는 점입니다.

# Numpy, Pandas 설치방법
- 두 패키지 모두 파이썬이 설치되어 있어야 합니다.

Window의 시작 버튼을 누르고, 명령 프롬프트(CMD)를 관리자 권한으로 실행한 뒤,

**pip install numpy**

**pip install pandas**를 입력합니다.

이후 파이썬을 실행해서(Window 시작화면에서 IDLE 검색한 뒤 실행)

**import numpy as np**

**import pandas as pd**를 실행하고, 오류 없이 정상적을 실행되면 설치가 완료되어 쓸 수 있습니다.

- import numpy as np를 입력한뒤, RuntimeError:가 뜹니다.
현재 numpy의 최신 버전이 윈도우 운영체제와 충돌하는 이슈가 있습니다.
최신버전인 1.19.4 버전 대신,

**pip uninstall numpy**

**pip install numpy==1.19.3**을 명령 프롬프트 창에서 실행하시면 정상적으로 설치가 완료됩니다.

# Numpy 배워보기

```python
import numpy as np
import pandas as pd
print (np.__version__)
print (pd.__version__)
```

    1.19.3
    1.1.5
    


```python
#np array 만들기
#여기서부터는 array = 배열로 취급
```
import numpy as np
import pandas as pd로 해당 python notebook에서 numpy와 pandas를 불러옵니다.

print (np._version_)은 해당 numpy의 버전을 출력합니다.

```python
one = [1,2,3,4,5]
print (one)
ten = [10,20,30,40,50]
print (ten)

type (one)
```

    [1, 2, 3, 4, 5]
    [10, 20, 30, 40, 50]
    




    list




```python```

one = [1,2,3,4,5], ten = [10,20,30,40,50]으로
one, ten 변수를 선언하고, 리스트를 저장합니다.

type(one)은 one의 타입을 출력합니다. list라고 나오게 됩니다.

```
onearr = np.array(one)
tenarr = np.array(ten)
print (onearr)
print (tenarr)

type (onearr)

```

    [1 2 3 4 5]
    [10 20 30 40 50]
    




    numpy.ndarray




```python```
```
one, ten 변수를 nd.array를 통해 onearr, tenarr에 numpy배열로 저장합니다.
이들을 출력하면 위의 리스트에서 ,가 빠진 값이 나옵니다.
```
onearr.dtype
```




    dtype('int32')




```python
```
다음은 onearr 변수의 자료형을 출력합니다. 자료형에는 int(signed), uint(unsigned), float, bool등이 있습니다.

int 오른쪽의 숫자는 비트를 뜻합니다. int32는 C언어의 int와 같은 자료형입니다.
int64를 쓰면 long long과 같은 자료형입니다.
```
onearr.shape
```




    (5,)




```python
```
다음은 onearr 배열의 크기를 확인할 수 있습니다. 5,는 5개의 값을 가진 1차원 배열이라는 뜻입니다.
```
arr2nd = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
arr2nd.shape
```




    (4, 3)




```python
```
이렇게 arr2nd배열을 2차원으로 만든뒤 shape를 입력하면 4 x 3의 2차원 배열이라는 것을 확인할 수 있습니다.
```
one.dtype
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-7-81347d222ef3> in <module>
    ----> 1 one.dtype
    

    AttributeError: 'list' object has no attribute 'dtype'



```python
sumarr = onearr + tenarr
sumlist = one + ten
print (sumarr)
print (sumlist)
```

    [11 22 33 44 55]
    [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    


```python
#기본 배열 만드는 함수 -> 빠르게 배열 만들기
```


```python
arr1 = np.zeros(3)
arr2 = np.zeros((5, 3))

print (arr1)
print (arr2)
```

    [0. 0. 0.]
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    


```python
arr1 = np.ones(3)
arr2 = np.ones((5, 3))

print (arr1)
print (arr2)
```

    [1. 1. 1.]
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    


```python
arr1 = np.identity(1)
arr2 = np.identity(2)

print (arr1)
print (arr2)
```

    [[1.]]
    [[1. 0.]
     [0. 1.]]
    


```python
arr1 = np.arange(100)
arr2 = np.arange(96,100)

print (arr1)
print (arr2)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
     24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
     48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
     72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
     96 97 98 99]
    [96 97 98 99]
    


```python
#기본적인 배열의 연산
```


```python
print (onearr)
print (tenarr)
```

    [1 2 3 4 5]
    [10 20 30 40 50]
    


```python
plus = onearr + tenarr
minus = onearr - tenarr
mul = onearr * tenarr
div = onearr / tenarr

print (plus)
print (minus)
print (mul)
print (div)
```

    [11 22 33 44 55]
    [ -9 -18 -27 -36 -45]
    [ 10  40  90 160 250]
    [0.1 0.1 0.1 0.1 0.1]
    


```python
arr1 = np.array([[10, 20, 30], [100, 200, 300]])
arr2 = np.array(([5, 6, 7], [5, 6, 7]))

print (arr1 + arr2)
print (arr1 - arr2)
```

    [[ 15  26  37]
     [105 206 307]]
    [[  5  14  23]
     [ 95 194 293]]
    


```python
arr1 = np.array([[10, 20], [30, 40]])
arr2 = np.identity(2)

print (arr1 * arr2)
```

    [[10.  0.]
     [ 0. 40.]]
    


```python
#numpy에서의 브로드캐스트
#numpy는 브로드 캐스트를 지원합니다. 몰론 두 array가 행이나 열의 길이가 같아야 합니다.
```


```python
arr1 = np.array([[10, 20],[30, 40],[50,60]])

print (arr1)
print (arr1.shape)
```

    [[10 20]
     [30 40]
     [50 60]]
    (3, 2)
    


```python
arr2 = np.array([100,200])
arr3 = np.array([[100], [200], [300]])

print (arr2.shape)
print (arr3.shape)
```

    (2,)
    (3, 1)
    


```python
horizontal = arr1 + arr2
print (horizontal)

vertical = arr1 + arr3
print (vertical)
```

    [[110 220]
     [130 240]
     [150 260]]
    [[110 120]
     [230 240]
     [350 360]]
    


```python
difarr = np.array([100,200,300,400])
sumarr2 = onearr + difarr
print (sumarr2)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-24-3603de6b54e0> in <module>
          1 difarr = np.array([100,200,300,400])
    ----> 2 sumarr2 = onearr + difarr
          3 print (sumarr2)
    

    ValueError: operands could not be broadcast together with shapes (5,) (4,) 



```python
#배열의 값 찾기
#파이썬, C언어에서 하는 배열 값 찾기와 비슷하다.

#1차원에서는 기본적으로 1개의 값만 입력하면 되지만, 2차원은 2개의 값을 입력한다.
```


```python
arr1 = np.arange(10)

print (arr1)
print (arr1[6])
print (arr1[0])
print (arr1[3:7])
print (arr1[:4])
print (arr1[6:])
print (arr1[:])
```

    [0 1 2 3 4 5 6 7 8 9]
    6
    0
    [3 4 5 6]
    [0 1 2 3]
    [6 7 8 9]
    [0 1 2 3 4 5 6 7 8 9]
    


```python
arr2 = np.array([[1,2,3],[4,5,6],[7,8,9]])

print (arr2)
print (arr2[2,2])
print (arr2[0,0])
print (arr2[0,:])
print (arr2[0:,1:])
print (arr2[:3,:1])
```

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    9
    1
    [1 2 3]
    [[2 3]
     [5 6]
     [8 9]]
    [[1]
     [4]
     [7]]
    


```python
#numpy의 함수들
#진짜 많다 여기 있는게 다가 아니다
```


```python
arr1 = np.random.random(10)
arr2 = np.random.randn(10)
arr3 = np.random.rand(10)
arr4 = np.random.randint(10)

print (arr1)
print (arr2)
print (arr3)
print (arr4)
```

    [0.9446711  0.83497248 0.83201387 0.13935773 0.17415421 0.49383566
     0.62534832 0.95301777 0.775304   0.42802872]
    [ 0.49760198  0.42642281 -0.17887658  0.97600137 -1.09890443  1.14944642
     -1.61367583  0.68313963 -0.69266217 -0.20675362]
    [0.45892907 0.79511622 0.23510859 0.83475971 0.61042409 0.98928205
     0.04701599 0.47026134 0.10542542 0.6393744 ]
    3
    


```python
np.random.rand(3, 10)
```




    array([[0.69049139, 0.33961382, 0.4720473 , 0.05657901, 0.44823818,
            0.62798771, 0.18857805, 0.58646854, 0.78296022, 0.23064811],
           [0.89095585, 0.56933846, 0.66368549, 0.36591346, 0.78302718,
            0.04110933, 0.23313533, 0.3403511 , 0.68102568, 0.99330884],
           [0.18974105, 0.01787721, 0.24513353, 0.61273492, 0.82943479,
            0.88117367, 0.39032357, 0.85654216, 0.57124692, 0.9269831 ]])




```python
arr1 = arr1 * 10

print (arr1)
print (np.ceil(arr1))
print (np.floor(arr1))
```

    [9.44671104 8.34972483 8.32013866 1.39357733 1.74154213 4.9383566
     6.2534832  9.53017769 7.75304001 4.28028724]
    [10.  9.  9.  2.  2.  5.  7. 10.  8.  5.]
    [9. 8. 8. 1. 1. 4. 6. 9. 7. 4.]
    


```python
arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])

print (np.sqrt(arr1))
```

    [[1.         1.41421356 1.73205081]
     [2.         2.23606798 2.44948974]
     [2.64575131 2.82842712 3.        ]]
    


```python
arr2 = np.array([[10, -100],[10, -1000]])

print (arr2)
print (np.abs(arr2))
```

    [[   10  -100]
     [   10 -1000]]
    [[  10  100]
     [  10 1000]]
    


```python
print (np.exp(arr1))
print (np.log10(arr1))
```

    [[2.71828183e+00 7.38905610e+00 2.00855369e+01]
     [5.45981500e+01 1.48413159e+02 4.03428793e+02]
     [1.09663316e+03 2.98095799e+03 8.10308393e+03]]
    [[0.         0.30103    0.47712125]
     [0.60205999 0.69897    0.77815125]
     [0.84509804 0.90308999 0.95424251]]
    


```python
print (np.cos(arr1))
```

    [[ 0.54030231 -0.41614684 -0.9899925 ]
     [-0.65364362  0.28366219  0.96017029]
     [ 0.75390225 -0.14550003 -0.91113026]]
    


```python
arr3 = np.array([[9,8,7],[6,5,4],[3,2,1]])

print (np.maximum(arr1, arr3))
print (np.minimum(arr1, arr3))
```

    [[9 8 7]
     [6 5 6]
     [7 8 9]]
    [[1 2 3]
     [4 5 4]
     [3 2 1]]
    


```python
#행렬의 정렬
```


```python
arr1 = np.random.rand(30)

arr1 = arr1 * 100
arr1 = np.floor(arr1)

print (arr1)

arr1 = arr1.astype('int32')

print (arr1)
print (arr1.dtype)
```

    [31. 75. 13. 88. 97. 83. 97. 52. 42. 40. 27. 39. 95. 78. 86.  9. 72. 85.
     55. 75.  8. 17. 77. 78. 55.  1. 81. 89. 62. 68.]
    [31 75 13 88 97 83 97 52 42 40 27 39 95 78 86  9 72 85 55 75  8 17 77 78
     55  1 81 89 62 68]
    int32
    


```python
print (np.sort(arr1))
print (np.sort(arr1)[::-1])
```

    [ 1  8  9 13 17 27 31 39 40 42 52 55 55 62 68 72 75 75 77 78 78 81 83 85
     86 88 89 95 97 97]
    [97 97 95 89 88 86 85 83 81 78 78 77 75 75 72 68 62 55 55 52 42 40 39 31
     27 17 13  9  8  1]
    

```python
```

# Pandas 배워보기

```python
import pandas as pd
import numpy as np
```


```python
data = {'name': ['YoungWoo', 'DomgHo', 'Minsu', 'Hong', 'Kwangsung'],
        'year': [2013, 2014, 2015, 2016, 2015],
        'points': [1.5, 1.7, 3.6, 2.4, 2.9]}

df = pd.DataFrame(data)
print (df)
```

            name  year  points
    0   YoungWoo  2013     1.5
    1     DomgHo  2014     1.7
    2      Minsu  2015     3.6
    3       Hong  2016     2.4
    4  Kwangsung  2015     2.9
    


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>year</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YoungWoo</td>
      <td>2013</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DomgHo</td>
      <td>2014</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Minsu</td>
      <td>2015</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hong</td>
      <td>2016</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kwangsung</td>
      <td>2015</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv = pd.read_csv('C:/Users/김영우/jupyter/example.csv')
csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>나이</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kim</td>
      <td>23</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Lee</td>
      <td>19</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Choi</td>
      <td>20</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Song</td>
      <td>23</td>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Hwang</td>
      <td>25</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (df.columns)
print (csv.columns)
```

    Index(['name', 'year', 'points'], dtype='object')
    Index(['ID', 'NAME', '나이', '점수'], dtype='object')
    


```python
print (df.values)
print (csv.values)
```

    [['YoungWoo' 2013 1.5]
     ['DomgHo' 2014 1.7]
     ['Minsu' 2015 3.6]
     ['Hong' 2016 2.4]
     ['Kwangsung' 2015 2.9]]
    [[1 'Kim' 23 75]
     [2 'Lee' 19 80]
     [3 'Choi' 20 59]
     [4 'Song' 23 90]
     [5 'Hwang' 25 83]]
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2014.600000</td>
      <td>2.420000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.140175</td>
      <td>0.864292</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2013.000000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2014.000000</td>
      <td>1.700000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.000000</td>
      <td>2.400000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2015.000000</td>
      <td>2.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2016.000000</td>
      <td>3.600000</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>나이</th>
      <th>점수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.000000</td>
      <td>5.00000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.000000</td>
      <td>22.00000</td>
      <td>77.400000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.581139</td>
      <td>2.44949</td>
      <td>11.631853</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>19.00000</td>
      <td>59.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>20.00000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>23.00000</td>
      <td>80.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>23.00000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.000000</td>
      <td>25.00000</td>
      <td>90.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['zero'] = np.zeros(5)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>year</th>
      <th>points</th>
      <th>zero</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YoungWoo</td>
      <td>2013</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DomgHo</td>
      <td>2014</td>
      <td>1.7</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Minsu</td>
      <td>2015</td>
      <td>3.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hong</td>
      <td>2016</td>
      <td>2.4</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kwangsung</td>
      <td>2015</td>
      <td>2.9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
csv['random'] = np.random.rand(5)
csv['random'] *= 10
csv
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>NAME</th>
      <th>나이</th>
      <th>점수</th>
      <th>random</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kim</td>
      <td>23</td>
      <td>75</td>
      <td>1.859625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Lee</td>
      <td>19</td>
      <td>80</td>
      <td>0.588391</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Choi</td>
      <td>20</td>
      <td>59</td>
      <td>8.702442</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Song</td>
      <td>23</td>
      <td>90</td>
      <td>7.638486</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Hwang</td>
      <td>25</td>
      <td>83</td>
      <td>9.592227</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df['zero']
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>year</th>
      <th>points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YoungWoo</td>
      <td>2013</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DomgHo</td>
      <td>2014</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Minsu</td>
      <td>2015</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hong</td>
      <td>2016</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kwangsung</td>
      <td>2015</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
val = pd.Series([10, 20, 30, 40, 50])
df['tens'] = val
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>year</th>
      <th>points</th>
      <th>tens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>YoungWoo</td>
      <td>2013</td>
      <td>1.5</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DomgHo</td>
      <td>2014</td>
      <td>1.7</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Minsu</td>
      <td>2015</td>
      <td>3.6</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Hong</td>
      <td>2016</td>
      <td>2.4</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kwangsung</td>
      <td>2015</td>
      <td>2.9</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>




```python
print (df.sum(axis = 0))
print (df.sum(axis = 1))
print (df.min(axis = 0))
```

    name      YoungWooDomgHoMinsuHongKwangsung
    year                                 10073
    points                                12.1
    tens                                   150
    dtype: object
    0    2024.5
    1    2035.7
    2    2048.6
    3    2058.4
    4    2067.9
    dtype: float64
    name      DomgHo
    year        2013
    points       1.5
    tens          10
    dtype: object
    


```python

```
