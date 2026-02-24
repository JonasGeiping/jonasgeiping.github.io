---
layout: page
permalink: /repositories/
title: Code
description: You can find implementations for a number of recent projects below.
nav: true
nav_order: 3
---

<!-- ## GitHub users

{% if site.data.repositories.github_users %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.html username=user %}
  {% endfor %}
</div>

---

{% if site.repo_trophies.enabled %}
{% for user in site.data.repositories.github_users %}
  {% if site.data.repositories.github_users.size > 1 %}
  <h4>{{ user }}</h4>
  {% endif %}
  <div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% include repository/repo_trophies.html username=user %}
  </div>

  ---

{% endfor %}
{% endif %}
{% endif %} -->

## GitHub Repositories

{% if site.data.repositories.github_repos %}
<div class="repositories d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.html repository=repo %}
  {% endfor %}
</div>

<script>
document.addEventListener("DOMContentLoaded", function() {
  document.querySelectorAll("[data-repo]").forEach(function(el) {
    var repo = el.getAttribute("data-repo");
    fetch("https://api.github.com/repos/" + repo)
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.description) el.textContent = data.description;
        var statsEl = document.querySelector('[data-repo-stats="' + repo + '"]');
        if (statsEl && data.stargazers_count !== undefined) {
          var parts = [];
          if (data.language) parts.push('<span>\u25cf ' + data.language + '</span>');
          parts.push('<span>\u2605 ' + data.stargazers_count + '</span>');
          if (data.forks_count) parts.push('<span>\u2442 ' + data.forks_count + '</span>');
          statsEl.innerHTML = parts.join('');
        }
      })
      .catch(function() {});
  });
});
</script>
{% endif %}
