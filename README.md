# ADI-FDTD — 3D 무조건 안정 전자기장 솔버

3차원 ADI-FDTD(교호 방향 음해법 FDTD) 솔버입니다.

* 텍스트 인풋 파일로 격자, 재료(ε_r, μ_r, σ, σ_m), 도체(PEC), PML, 포트를 지정합니다.
* 구조물은 **외부 비정렬 메쉬 파일**(Gmsh `.msh`, STL)에서 읽을 수 있습니다.
* 격자는 **비균일 직교 격자**로, 구조 주변만 국소 세분화할 수 있습니다.
* 포트에 전기장을 가해 여기하고, 출력은 E·H 필드 GIF 애니메이션과 포트 전압·전류(S11)입니다.

```
pip install -e .                                   # 설치 (numpy, matplotlib, pillow)
python -m adifdtd examples/dipole.txt              # 실행
python -m adifdtd input.txt -o result_dir          # 출력 폴더 지정
python -m adifdtd.postproc result_dir/ports.csv    # S11 계산 (png, csv)
python tests/test_solver.py; python tests/test_mesh.py
```

| 패치 안테나 (Gmsh 메쉬, 비균일 격자) | S11 |
|---|---|
| ![patch](docs/patch_Ez_substrate_xy.gif) | ![s11](docs/patch_S11.png) |

---

## 1. 수치 기법

| 항목 | 내용 |
|---|---|
| 시간 적분 | Zheng/Namiki ADI (Peaceman–Rachford) 2-서브스텝. 격자선 방향 3중대각 행렬을 Thomas 알고리즘으로 풂 (LU 사전 분해) |
| 안정성 | Δt에 무관하게 안정. 비균일 격자 + ε/μ/σ/σ_m + PML + 포트 조합에서 증폭행렬 스펙트럼 반경 = 1 확인 (CFL 200까지) |
| 격자 | 비균일 직교(rectilinear) Yee 격자. 외곽 경계는 PEC (PML 뒤쪽) |
| 재료 | 셀 단위 ε_r, μ_r, σ(전기 손실), σ_m(자기 손실). E edge는 인접 4셀 면적 가중 평균, H face는 인접 2셀 조화 평균(μ) |
| 구조 입력 | 박스(격자 인덱스 또는 물리 좌표) + Gmsh/STL 메쉬 영역 |
| PML | 등급형 임피던스 정합 손실층 (σ_e/ε = σ_m/μ, 유전체·자성체 내부에서도 정합) |
| 포트 | 내부저항 R의 lumped 저항성 전압원. 정합 R이면 되돌아온 파를 흡수 |

### 비정렬 메쉬와 격자의 관계
ADI는 격자선 방향으로 3중대각 방정식을 푸는 방식이라 **직교 격자가 필수**입니다. 사면체 격자 위에서 직접 시간 적분할 수 없습니다.
그래서 이 코드는 다음 구조를 씁니다.

1. 구조물은 CAD/Gmsh에서 만든 **비정렬 메쉬**(사면체·삼각형·선 요소)로 정의합니다.
2. 솔버가 메쉬를 **비균일 직교 FDTD 격자**에 매핑합니다.
   * **부피 영역**(사면체, 닫힌 STL): 경계면을 추출해 z방향 ray-parity로 내부를 판정합니다. 셀마다 `subcell³`개 샘플로 **부피 분율**을 구하고 ε, μ, σ를 분율로 혼합합니다(경계 계단 오차 감소). PEC 부피는 면·모서리 위 edge까지 포함해 E=0으로 둡니다.
   * **면 영역**(삼각형): 얇은 PEC 판입니다. 샘플점을 가장 가까운 노드에 스냅하고, 두 노드가 모두 판 위인 edge를 PEC로 둡니다.
   * **선 영역**(선 요소): 얇은 PEC 와이어입니다. 계단형 경로로 스냅합니다.
3. 얇은 구조나 경계 근처는 `xlines/ylines/zlines`로 국소 세분화합니다. 이렇게 해도 무조건 안정성은 유지되므로 Δt는 가장 작은 셀이 아니라 관심 주파수 기준으로 잡을 수 있습니다.

### PML 방식
CPML(stretched coordinate)과 split-field PML은 ADI와 결합하면 CFL > 1에서 지수적으로 발산했습니다 (증폭행렬 |λ| ≈ 1.02–1.3). 그래서 손실을 대각항으로만 넣는 정합 손실형 PML을 사용합니다. 각 서브스텝 연산자가 "반대칭 + 소산"이 되어 무조건 안정이 보장됩니다.

* `uniaxial`(기본): 축 d 방향 층에서 접선 성분(c ≠ d)에만 손실을 줍니다. 수직 입사 무반사입니다.
* `isotropic`: 모든 성분에 손실을 줍니다.

성능 (8셀, 대형 도메인 기준해 대비): CFL 2에서 −45 ~ −50 dB, CFL 5(주기당 3.5 스텝)에서 −22 ~ −30 dB. **주기당 10 스텝 이상**을 권장합니다.

### 검증 결과
| 항목 | 결과 |
|---|---|
| 무조건 안정성 | 비균일 격자, ε/μ/σ/σ_m, PEC, 포트, PML 조합에서 스펙트럼 반경 1.000000 (CFL 30, 200) |
| μ 구현 | 평행판 선로에 매질 삽입. ε=4와 μ=4의 반사 크기 동일(쌍대성). ε=μ=4(임피던스 정합)는 반사 0.02. 지연시간은 n=2, 4 이론과 ±2 스텝 일치 |
| 비균일 격자 | 외곽만 1.8배 등급 격자 vs 균일 격자: 프로브 신호 차이 1.1% |
| 메쉬 매핑 | 반지름 8 mm 구 STL 부피 오차 0.7%. 사면체 큐브 부피 정확 |
| 패치 안테나 | FR-4 패치 공진 2.97 GHz (S11 −12.6 dB). 전송선로 모델 3.18 GHz 대비 −7% (PEC 판 가장자리 계단 근사의 전형적 차이) |
| 포트 반사 억제 | 정합 R에서 단락 끝 2차 에코 약 −20 dB, 부정합(10배 R)에서는 반복 에코 |

---

## 2. 인풋 파일 형식

한 줄에 `키워드 위치인자... key=value...` 형식으로 씁니다. `#` 뒤는 주석입니다. 단위는 SI(m, s, S/m)입니다.

### 격자
| 키워드 | 형식 | 설명 |
|---|---|---|
| `grid` + `cell` (+ `origin`) | `Nx Ny Nz` / `dx [dy dz]` / `x0 y0 z0` | 균일 격자 |
| `xlines` `ylines` `zlines` | `x0  x1:n1[:r1]  x2:n2[:r2] ...` | 비균일 축. 구간 [x_{k-1}, x_k]를 n_k개 셀로 나눔. r_k = 마지막 셀/첫 셀 크기비 (등비 등급, 기본 1) |
| `cfl` | `값` | Δt = 값 × 가장 작은 셀 기준 Yee CFL 한계 (1보다 커도 됨) |
| `dt` | `초` | Δt 직접 지정 |
| `steps` | `N` | 시간 스텝 수 |

축마다 섞어 쓸 수 있습니다. 예: x, y는 `grid`/`cell`로 균일하게, z만 `zlines`로 둘 수 있습니다.
인접 셀 크기비는 2 이하를 권장합니다.

```
xlines -0.050  -0.022:12:0.25  0.022:44  0.050:12:4    # 외곽 4mm → 중앙 1mm → 외곽 4mm
zlines -0.030  -0.002:10:0.2   0.0:4  0.0016:4  0.004:5  0.032:10:5
```

### PML
| 키워드 | 형식 | 설명 |
|---|---|---|
| `pml` | `n` / `nx ny nz` / `xmin xmax ymin ymax zmin zmax` | 각 면 PML 두께(셀). 옵션 `type= order= r0=` |
| `pml_type` / `pml_order` / `pml_reflection` | | `uniaxial`\|`isotropic` / 등급 차수 m (기본 3) / 이론 반사계수 R0 (기본 1e-4) |

### 박스 구조물
박스 인덱스는 기본이 **격자 노드 번호** `i0 i1 j0 j1 k0 k1`입니다. `coords=phys`를 붙이면 **물리 좌표 [m]**로 읽고 가장 가까운 노드로 맞춥니다.

| 키워드 | 형식 |
|---|---|
| `conductor` | `i0 i1 j0 j1 k0 k1 [coords=phys]` — PEC 박스 (한 축이 같으면 판, 두 축이 같으면 선) |
| `material` (=`dielectric`) | `i0 i1 j0 j1 k0 k1 eps= mu= sigma= sigma_m= [coords=phys]` |

### 메쉬 구조물
```
mesh   file=patch.msh scale=1e-3 offset=0,0,0 subcell=2
region substrate type=material eps=4.4 mu=1 sigma=0.015 sigma_m=0
region ground    type=pec
region 4         type=pec
stl    file=sphere.stl scale=1e-3 offset=0.005,0,0 type=material eps=2 mu=3 subcell=3
```
* `mesh` : 메쉬 파일을 읽습니다 (경로는 인풋 파일 기준 상대경로). `scale`은 메쉬 단위를 m로 바꾸는 배율, `offset`은 이동량[m], `subcell`은 부피 분율 샘플 수(축당)입니다.
* `region <물리그룹 번호 또는 이름>` : 바로 위 `mesh`의 물리그룹에 재료를 지정합니다.
  * `type=material` : 부피 영역만 가능합니다. `eps`, `mu`, `sigma`, `sigma_m`를 지정합니다.
  * `type=pec` : 부피(속이 찬 도체), 면(얇은 판), 선(와이어) 모두 가능합니다.
* `stl` : 닫힌 STL 곡면 하나를 부피 영역으로 읽습니다. 같은 줄에 `type`과 재료를 씁니다.
* 지원 형식: Gmsh `.msh` ASCII v2.2/v4.1 (1·2차 선, 삼각형, 사면체), STL ASCII/binary.
* 적용 순서: 인풋 파일 순서대로 덮어씁니다. PEC는 항상 재료보다 우선합니다.

### 포트
```
port 0 0 -0.005 -0.005 0 0.0004 coords=phys dir=+z amp=2500 R=50 wave=modgauss f0=3.2e9 tau=2.5e-10 name=feed
```
* `dir` : 인가 전기장 방향 `±x, ±y, ±z`. 박스는 이 축으로 1셀 이상 길이가 있어야 합니다.
* `amp` : 인가 전기장 세기 [V/m]. 개방 기전력 = amp × 포트 길이.
* `R` : 내부저항 [Ω]. 선로 임피던스와 맞추면 반사파를 흡수합니다.
* 단면 방향으로 넓히면 sheet 포트가 됩니다.
* 파형:
  * `gauss` (`tau`, `t0`)
  * `modgauss` (`f0`, `tau`, `t0`)
  * `sin` (`f0`, `ramp`)
  * `ricker` (`f0`, `t0`)

### 출력
```
output E comp=z   plane=xy pos=0.0008 interval=8 file=Ez.gif
output H comp=mag plane=xz index=20   interval=5 fps=15 clim=2.0 gridlines=yes
```
* `comp` : `x`, `y`, `z`, `mag`
* `plane` : `xy`, `xz`, `yz`
* 평면 위치는 `pos=`(물리 좌표 [m]) 또는 `index=`(셀 번호)로 지정합니다.
* `interval` : GIF 저장 간격(스텝)
* `clim` : 컬러 범위 고정 (기본: 99.5 백분위)
* `gridlines=yes` : 비균일 격자선을 표시합니다.
* GIF 축은 물리 좌표[mm]입니다. 도체(검정), 재료(연두), 포트(하늘색), PML 경계(회색 점선)를 표시합니다.

포트가 있으면 `ports.csv`(t, E_s, 평균 E, 전압 V, 전류 I)를 저장합니다. `python -m adifdtd.postproc` 명령으로 S11을 계산할 수 있습니다.

---

## 3. 예제

| 폴더 / 파일 | 내용 |
|---|---|
| `examples/mesh_patch/` | FR-4 프로브 급전 패치 안테나. `patch.msh`(Gmsh)에서 기판(사면체), 접지·패치(삼각형 판), 급전핀(선)을 읽음. 비균일 격자(기판 0.4 mm, 중앙 1 mm, 외곽 4 mm), CFL 4.3. 약 1.5분 |
| `examples/mesh_sphere/` | `sphere.stl`(ε_r=2, μ_r=3 자성 유전체 구) + 다이폴 산란. 박스·포트를 물리 좌표로 지정 |
| `examples/dipole.txt` | 갭 급전 다이폴 + 유전체 블록 (균일 격자, 인덱스 지정) |
| `examples/parallel_plate_short.txt` | 평행판 선로 + 정합 sheet 포트 + 단락 (반사 흡수 확인) |

메쉬 재생성: `pip install gmsh` 후 `python examples/mesh_patch/make_mesh.py` 실행. 또는 Gmsh GUI에서 `patch.geo`를 열어 ASCII `.msh`로 내보냅니다.

```
cd examples/mesh_patch
python -m adifdtd patch_antenna.txt
python -m adifdtd.postproc output_patch/ports.csv --fmin 1.5e9 --fmax 5e9
```

## 4. 코드 구성

```
adifdtd/
  inputfile.py   인풋 파서, 설정 데이터 구조
  grid.py        비균일 직교 격자 (노드 좌표, primal/dual 간격)
  meshfile.py    Gmsh .msh(v2.2/v4.1)·STL 리더
  geometry.py    메쉬/박스 → 격자 매핑 (ray-parity 복셀화, 부피 분율, 판·와이어 스냅)
  solver.py      ADI-FDTD 코어 (계수, 3중대각 분해, 서브스텝 갱신, 포트)
  waveforms.py   포트 여기 파형
  output.py      평면 슬라이스 샘플링, GIF 작성
  postproc.py    ports.csv → S11
tests/           안정성(고유값), 메쉬 리더·복셀화 테스트
```
