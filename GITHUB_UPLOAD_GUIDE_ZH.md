# GitHub 发布步骤

本目录已经按科研复现仓库整理。公开版本包含代码、汇总结果和复现说明，但不包含分析工作簿。数据获取和本地放置方式见 `data/README.md`。

## 一、上传前检查

在 PowerShell 中进入本目录：

```powershell
cd "<path-to-repository>\ads-shadow-credal-xai"
python scripts/verify_release.py
python -m pytest -q
```

两条命令都通过后再继续。不要把 `outputs/`、检查点、崩溃转储、虚拟环境或 `data/ads_crashes.xlsx` 加入仓库；`.gitignore` 已排除这些内容。

## 二、在 GitHub 创建空仓库

1. 打开 <https://github.com/>，使用目标账户绑定的邮箱登录。
2. 单击右上角 `+`，选择 `New repository`。
3. Repository name 填写 `ads-shadow-credal-xai`。
4. Description 可填写：`Reference implementation of a shadow-calibrated multi-explainer framework for specification-robust predictive relevance, credal partial ordering, and complementary PAG-based structural interpretation, demonstrated on ADS crash severity data.`
5. Visibility 选择 `Public` 前，确认分析工作簿及其他受限材料未被跟踪。
6. 不勾选 `Add a README file`，不添加 `.gitignore`，也暂不选择 license。本地目录已经包含前两项；许可证应由权利人与专利代理人共同确认。
7. 单击 `Create repository`。

注意：邮箱地址是登录凭据，不一定是 GitHub 用户名。创建完成后，浏览器地址栏中 `github.com/` 后面的字符串才是远程地址所需的用户名。

## 三、连接已准备好的本地仓库

本发布目录已经初始化、提交并打好版本标签，因此不要再次执行 `git init`、`git add .` 或网页中显示的“create a new repository”命令。上传前可执行：

```powershell
git status
git ls-files data
```

`git status` 应显示工作区干净；`git ls-files data` 只应显示 `data/README.md`，不应显示 `data/ads_crashes.xlsx`。

## 四、连接远程仓库并上传

把 `<github-username>` 替换为网页中显示的实际用户名：

```powershell
git remote add origin https://github.com/<github-username>/ads-shadow-credal-xai.git
git remote -v
git push -u origin main
git push origin v1.0.0
```

首次推送通常会由 Git Credential Manager 打开浏览器。登录并授权后返回终端等待完成。GitHub 已停止接受账户密码进行 Git 操作；如果浏览器授权没有出现，应使用 fine-grained personal access token，而不是把账户密码输入终端。

创建 token 的路径为 `GitHub Settings > Developer settings > Personal access tokens > Fine-grained tokens`。只授权刚创建的仓库，并赋予 `Contents: Read and write`。token 只在生成时显示一次，不要写进代码、README、命令脚本或聊天记录。

## 五、上传后核对

打开仓库网页，逐项确认：

1. 仓库为 `Public`，且未包含分析工作簿或其他受限数据。
2. 首页正确显示 `README.md`。
3. `data/README.md`、`analysis/`、`figures/`、`results/` 和 `provenance/` 均存在。
4. `Actions` 页面中的 `repository-qa` 运行通过。
5. 仓库中没有个人桌面路径、原始事故编号、临时检查点、模型缓存或转储文件。

## 六、后续更新

每次修改后依次执行：

```powershell
python scripts/verify_release.py --write-checksums
python scripts/verify_release.py
python -m pytest -q
git add .
git commit -m "Describe the change briefly"
git push
```

在论文接收后，建议为正式版本创建 GitHub Release，并将其归档至 Zenodo 以获得 DOI。公开许可、论文 DOI 与最终作者顺序也应在此时更新到 `CITATION.cff`。
