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
**pip install numpy
pip install pandas**를 입력합니다.

이후 파이썬을 실행해서(Window 시작화면에서 IDLE 검색한 뒤 실행)
**import numpy as np
import pandas as pd**를 실행하고, 오류 없이 정상적을 실행되면 설치가 완료되어 쓸 수 있습니다.

- import numpy as np를 입력한뒤, RuntimeError:가 뜹니다.
현재 numpy의 최신 버전이 윈도우 운영체제와 충돌하는 이슈가 있습니다.
최신버전인 1.19.4 버전 대신,
**pip uninstall numpy
pip install numpy==1.19.3**을 명령 프롬프트 창에서 실행하시면 정상적으로 설치가 완료됩니다.

# Numpy 배워보기


