# 发布到PyPI指南

本文档详细说明如何将此Python项目发布到PyPI（Python Package Index）。

## 准备工作

1. 在PyPI和TestPyPI上创建账户：
   - [TestPyPI](https://test.pypi.org/account/register/)（用于测试）
   - [PyPI](https://pypi.org/account/register/)（生产环境）

2. 安装必要的工具：
   ```bash
   pip install build twine
   ```

## 构建项目

1. 确保项目版本号在[pyproject.toml](file://C:\code\mcp\mcp_study\weather001\pyproject.toml)中已正确设置：
   ```toml
   [project]
   version = "0.1.0"
   ```

2. 清理之前的构建文件（如果有）：
   ```bash
   rm -rf dist/
   ```

3. 构建分发包：
   ```bash
   python -m build
   ```

   这将创建两个文件：
   - `dist/weather001-0.1.0.tar.gz`（源代码分发）
   - `dist/weather001-0.1.0-py3-none-any.whl`（wheel分发）

## 上传到TestPyPI（推荐）

在正式发布到PyPI之前，建议先上传到TestPyPI进行测试：

```bash
twine upload -r testpypi dist/*
```

系统会提示您输入TestPyPI的用户名和密码。

## 安装和测试TestPyPI版本

上传后，您可以从TestPyPI安装和测试您的包：

```bash
pip install --index-url https://test.pypi.org/simple/ weather001
```

## 上传到PyPI

确认TestPyPI版本工作正常后，可以上传到正式的PyPI：

```bash
twine upload dist/*
```

系统会提示您输入PyPI的用户名和密码。

## 使用API令牌（推荐）

为了安全起见，建议使用API令牌而不是用户名/密码：

1. 在PyPI上生成API令牌：
   - 登录PyPI
   - 转到"Account settings"
   - 选择"API tokens"
   - 点击"Add API token"

2. 使用令牌上传：
   ```bash
   twine upload -u __token__ -p YOUR_API_TOKEN dist/*
   ```

## 版本管理

每次更新项目时，请记得更新[pyproject.toml](file://C:\code\mcp\mcp_study\weather001\pyproject.toml)中的版本号：

```toml
[project]
version = "0.1.1"  # 增加版本号
```

遵循语义化版本控制（SemVer）：
- MAJOR版本：不兼容的API更改
- MINOR版本：向后兼容的功能新增
- PATCH版本：向后兼容的错误修复

## 验证发布

发布后，您可以在PyPI上查看您的项目：
https://pypi.org/project/weather001/

用户现在可以通过以下命令安装您的包：
```bash
pip install weather001
```

或者使用uvx运行：
```bash
uvx weather001
```