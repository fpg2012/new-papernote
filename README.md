# 论文笔记

个人论文阅读笔记，全部为 Markdown 文件，格式：`YYYY-MM-DD_作者_标题.md`，存放于 [`docs/`](docs/) 目录。

- push 到 `main` 分支后，GitHub Actions 会自动用 [MkDocs](https://www.mkdocs.org/) + [Material 主题](https://squidfunk.github.io/mkdocs-material/) 将 md 构建为 HTML 并发布到 GitHub Pages。
## 新增笔记

在 `docs/` 下添加 md 文件并提交即可，站点自动更新。

### front matter 规范（YAML 必须合法）

```yaml
---
领域: <研究领域>
发表时间: <论文发表时间>
读文章时间: <阅读日期>
标题: <论文标题，含冒号时请加引号，如 "Title: Subtitle">
链接: https://arxiv.org/abs/xxxx
Zotero: <入库状态>（可选）
页码说明: <页码约定>（可选）
---
```

front matter 中的 `领域/发表时间/读文章时间/链接/Zotero/页码说明` 会显示在页面标题下方。
