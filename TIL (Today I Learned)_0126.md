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
