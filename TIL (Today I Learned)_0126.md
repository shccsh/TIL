# TIL (Today I Learned)_Github



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
   - `git log --oneline` 을 기입하면 해쉬값을 알려준다. 그걸 아래 체크아웃에 쓰기로 하자.
6. `git checkout 해쉬값` : 돌아가기
   - `git checkout master` : 다시 마스터로 돌아올 때
   - `git checkout head~3` : 현재에서 몇개 뒤로 돌아가라
7. `git remote add origin 주소` : 브릿지 잇기
   - `git remote - v` : 이어진거 확인
8. `git push origin master` : 올리기
9. `.gitignore` : 특정 파일 혹은 폴더에 대해 Git이 버전 관리를 하지 못하도록 지정하는 것
   - `touch .gitignore` 로 생성
   - [gitignore](https://gitignore.io/) 사이트에서 원하는 부분을 검색하여 쉽게 작성할 수 있다.
10. `git clone` : 원격 저장소의 커밋 내역을 모두 가져와서, 로컬 저장소를생성하는 명령어
    - 결과적으로 `git init` 과 같은 결과를 가져오나, 방식은 반대 (hub에 있는 것을 내 PC 내 복제하는 것)
    - `git clone <원격 저장소 주소>` 형태로 입력
    - 생성된 로컬 저장소는 `git init` 과 `git remote add` 가 이미 수행되어 있다.
11. `git pull` : 원격 저장소의 변경 사항을 가져와서, 로컬 저장소를 업데이트하는 명령어
    - `git pull <저장소 이름> <브랜치 이름>` 형태로 입력

### 1) clone, pull, push Relay-test

1. 규칙

   - 두개의 컴퓨터가 있다고 가정(다른 공간)

     ex) acomputer(집 컴퓨터), bcomputer(회사 컴퓨터)

   - 집에서 작업을 하다, 회사에서 작업을 하는 상황

2. 집에서 작업하고 깃허브에 올리기 

   - Github에서 Relay-test 이름의 `Repositories(원격저장소)`생성

   - `acomputer 폴더` 에서  vscode를 열고, 아래와 같은 절차 진행

     ``````bash
     $ git init                      # 관리 시작
     $ touch a.txt                   # a.txt 문서 생성
     $ git add .                     # 로컬에서 working directory에서 staging Area로 올림
     $ git commit -m "집에서 작성"     # 커밋(버전)으로 저장
     $ git remote add origin https://github.com/shccsh/relay-test.git   # 아래 이미지 영역에서 URL을 받아 브릿지 잇기
     $ git push origin master        # Github에 올리기
     ``````

     ![image-20220127114429489](TIL (Today I Learned)_0126.assets/image-20220127114429489.png)

   - 위의 절차를 거치면, hub에 a. txt가 올라가지게 된다.

3. 회사에서 집에서 작업한 내용을 복제하기

   - 홈폴더 또는 작업하고자 하는 폴더에서 Bash를 키고 아래 내용을 입력하여 복제한다.

     ``````bash
     $ git clone https://github.com/shccsh/relay-test.git
     ``````

   - 복제하면 `Relay-test` 폴더가 만들어지고 이를 bcomputer로 이름을 수정한다.(위의 규칙)

   - 복제된 폴더는 init과 remote 브릿지 연결까지 되어있다.

   - 복제할 때 마지막에 폴더명을 넣으면 해당 폴더명으로 생성되기도 한다.

     ``````bash
     $ git clone https://github.com/shccsh/relay-test.git bcomputer
     ``````

4. 회사에서 추가 작업하기

   - 3번에서 만들어진 폴더에서 vscode를 열고, 작업을 진행한다.

   - a.txt를 수정하는 작업 진행 후, 2번의 작업을 다시 진행

     ``````bash
     # a.txt 수정 후,
     $ git add .
     $ git commit -m "회사에서 작성"
     $ git push origin master
     ``````

5. 집에서 회사에서 작업한 내용 불러오기

   - 집에서 작업한 로컬과 회사에서 작업해서 hub에 올라간 내용은 다르기 때문에, hub에 올라간 내용을 불러와야한다.

   - 불러올 때는 `pull`을 사용한다.

     ``````bash
     $ git pull origin master
     ``````

   - 이어서 추가 작업을 진행 후, 다시 4번의 올리는 작업을 진행한다.

### 2) 충돌

1. 발생 원인

   - 로컬이 아닌 원격저장소에서 직접 수정을 하고, 로컬에서 수정을 한 내용을 올릴 경우 충돌이 발생한다.

   - 로컬의 내용과 원격저장소의 내용이 다른 상황에서 작업이 이뤄지기 때문에 발생

     ![image-20220127122915297](TIL (Today I Learned)_0126.assets/image-20220127122915297.png)

2. 해결 방법

   - `$ git pull origin master` 를 입력

   - 저장소에서 작업한 내용, 로컬에서 작업한 내용이 같이 보여진다.

     ![image-20220127122951691](TIL (Today I Learned)_0126.assets/image-20220127122951691.png)

   - 원하는 것을 선택해서 하나만 보여지게 해도 되고, 둘다 보여지게 해도 되고, 새롭게 수정을 해도 된다.

   - 그리고, 현재 시점에는 아래와 같이 터미널 부분이 master|MERGING 으로 된 것을 알 수 있다.

     ![image-20220127123236508](TIL (Today I Learned)_0126.assets/image-20220127123236508.png)

   - 수정을 하고 나서 다시 `git add .` , `git commit -m "메시지"` 를 하면 정상적으로 돌아오게 된다.

     ![image-20220127123527923](TIL (Today I Learned)_0126.assets/image-20220127123527923.png)

   





