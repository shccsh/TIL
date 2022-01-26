# TIL (Today I Learned)_0126



## 1. 로컬 저장소

- `Working Directory (=Working Tree)` : 사용자의 일반적인 작업이 일어나는 곳
- `Staging Area (=Index)` : 커밋을 위한 파일 및 폴더가 추가되는 곳
- `Repository` : staging area에 있던 파일 및 폴더의 변경사항(커밋)을 저장하는 곳
- Git은 Working Directory - Staging Area - Repository의 과정으로 버전 관리를 수행



## 2. git 명령어

1. `git init` : 현재 폴더를 깃이 관리하는 폴더로 만들어줘!
   - 홈폴더에서 기입하면 안된다.(너무 많은 데이터가 있기 때문)
   - 최초 1번만 기입한다.
   - VS code의 베쉬 부분 입력하면 맨 우측에 (master)가 뜬다.

2. `git status` : 현 상황을 보고 싶어!
   - Working Directory와 Staging Area에 있는 파일의 현재 상태를 알려주는 명령어
   - 어떤 작업 시행하기 전에 수시로 status 확인하는 것이 좋다고 한다.
3. `git add a.txt` : a.txt 올리기
   - `git add .` : 전부 다 올리기
   - Working Directory에 있는 파일을 Staging Area로 올리는 명령어

4. `git commit -m "메시지"` : 찰칵! 후 저장소
   - Staging Area에 올라온 파일의 변경 사항을 하나의 버전(커밋)으로 저장하는 명령어
   - ``커밋 메세지``는 현재 변경 사항들을 잘 나타낼 수 있도록 의미있게 작성하는 것을 권장

5. `git log` : 버전들 확인할래!
   - 커밋의 내역(`ID, 작성자, 시간, 메세지 등`)을 조회할 수 있는 명령어

