# Webpage Update for Dummies (me)

Local testing:
```bash
bundle config set --local path 'vendor/bundle'

bundle exec jekyll serve --lsi
```

Adjust bibliography:
1. Autogen paper.bib file via zotero.
2. Add `selected = {true}`, to bib entries that are selected publications
3. Add pictures (maybe), by adding them to `assets/img/publication_preview/`, with a matching key to zotero, and add `preview={picture.png}`.

4. Make sure github actions work as expected and deployment runs.
