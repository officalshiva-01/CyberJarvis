from typing import Dict, Any, List
from integrations.api_client import BaseApiClient
from config import GITHUB_TOKEN
from tools.registry import registry
from security.policy import RiskLevel, PermissionLevel


class GitHubClient(BaseApiClient):
    def __init__(self, token: str = GITHUB_TOKEN):
        super().__init__(base_url="https://api.github.com")
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def get_repo(self, owner: str, repo: str) -> Dict[str, Any]:
        res = self.get(f"/repos/{owner}/{repo}", headers=self._headers())
        if "error" in res:
            return res
        return {
            "name": res.get("full_name"),
            "description": res.get("description"),
            "stars": res.get("stargazers_count"),
            "forks": res.get("forks_count"),
            "open_issues": res.get("open_issues_count"),
            "default_branch": res.get("default_branch"),
            "html_url": res.get("html_url"),
        }

    def get_commits(self, owner: str, repo: str, limit: int = 5) -> List[Dict[str, Any]]:
        res = self.get(f"/repos/{owner}/{repo}/commits", params={"per_page": limit}, headers=self._headers())
        if isinstance(res, dict) and "error" in res:
            return [res]
        if not isinstance(res, list):
            return [{"error": "Unexpected GitHub response format"}]

        commits = []
        for c in res[:limit]:
            commits.append({
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name"),
                "date": c.get("commit", {}).get("author", {}).get("date"),
            })
        return commits


github_client = GitHubClient()


@registry.register(
    name="get_github_repo_info",
    description="Fetch repository details and statistics from GitHub.",
    parameters={
        "owner": {"type": "string", "description": "Repository owner / organization", "required": True},
        "repo": {"type": "string", "description": "Repository name", "required": True},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_github_repo_info(owner: str, repo: str) -> Dict[str, Any]:
    return github_client.get_repo(owner, repo)


@registry.register(
    name="get_github_recent_commits",
    description="Fetch latest commits for a remote GitHub repository.",
    parameters={
        "owner": {"type": "string", "description": "Repository owner", "required": True},
        "repo": {"type": "string", "description": "Repository name", "required": True},
        "limit": {"type": "integer", "description": "Max commits (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
)
def get_github_recent_commits(owner: str, repo: str, limit: int = 5) -> List[Dict[str, Any]]:
    return github_client.get_commits(owner, repo, limit=limit)
