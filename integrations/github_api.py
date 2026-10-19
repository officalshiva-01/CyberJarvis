from typing import Dict, Any, List, Union
from integrations.api_client import BaseApiClient, redact_secrets
from config import GITHUB_TOKEN
from tools.registry import registry
from tools.schemas import IsolationMode
from security.policy import RiskLevel, PermissionLevel

DEFAULT_OWNER = "officialshiva-01"
DEFAULT_REPO = "CyberJarvis"


class GitHubClient(BaseApiClient):
    """Secure client for GitHub REST API v3 with authentication and secret protection."""

    def __init__(self, token: str = GITHUB_TOKEN):
        super().__init__(base_url="https://api.github.com")
        self.token = token

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        return headers

    def get_repo(self, owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> Dict[str, Any]:
        """Fetch repository details from GitHub API."""
        owner = owner or DEFAULT_OWNER
        repo = repo or DEFAULT_REPO
        res = self.get(f"/repos/{owner}/{repo}", headers=self._headers())
        if isinstance(res, dict) and "error" in res:
            return res

        return {
            "name": res.get("full_name", f"{owner}/{repo}"),
            "owner": owner,
            "repo": repo,
            "description": res.get("description") or "No description provided.",
            "stars": res.get("stargazers_count", 0),
            "forks": res.get("forks_count", 0),
            "open_issues": res.get("open_issues_count", 0),
            "default_branch": res.get("default_branch", "main"),
            "html_url": res.get("html_url", f"https://github.com/{owner}/{repo}"),
            "language": res.get("language", "Python"),
            "visibility": res.get("visibility", "public"),
            "success": True,
            "verified": True,
        }

    def get_commits(self, owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch latest commits for a remote GitHub repository."""
        owner = owner or DEFAULT_OWNER
        repo = repo or DEFAULT_REPO
        limit = max(1, min(int(limit), 20))
        res = self.get(f"/repos/{owner}/{repo}/commits", params={"per_page": limit}, headers=self._headers())

        if isinstance(res, dict) and "error" in res:
            return [res]
        if not isinstance(res, list):
            return [{"error": "Unexpected GitHub response format", "status": "invalid_response", "success": False}]

        commits = []
        for c in res[:limit]:
            commits.append({
                "sha": c.get("sha", "")[:7],
                "message": c.get("commit", {}).get("message", "").split("\n")[0],
                "author": c.get("commit", {}).get("author", {}).get("name", "Unknown"),
                "date": c.get("commit", {}).get("author", {}).get("date", ""),
            })
        return commits

    def get_status(self, owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> Dict[str, Any]:
        """Fetch combined repository health and status summary."""
        repo_data = self.get_repo(owner, repo)
        if "error" in repo_data:
            return repo_data

        commits = self.get_commits(owner, repo, limit=1)
        latest_commit = commits[0] if (commits and "error" not in commits[0]) else {}

        return {
            "name": repo_data.get("name"),
            "owner": owner,
            "repo": repo,
            "description": repo_data.get("description"),
            "stars": repo_data.get("stars", 0),
            "forks": repo_data.get("forks", 0),
            "open_issues": repo_data.get("open_issues", 0),
            "default_branch": repo_data.get("default_branch", "main"),
            "html_url": repo_data.get("html_url"),
            "latest_commit": latest_commit,
            "status": "active",
            "success": True,
            "verified": True,
        }


github_client = GitHubClient()


# -------------------------------------------------------------
# Registered Tool Definitions
# -------------------------------------------------------------

@registry.register(
    name="get_repository",
    description="Fetch GitHub repository metadata, description, stars, forks, and default branch.",
    parameters={
        "owner": {"type": "string", "description": "Repository owner / organization", "default": DEFAULT_OWNER},
        "repo": {"type": "string", "description": "Repository name", "default": DEFAULT_REPO},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_repository(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> Dict[str, Any]:
    return github_client.get_repo(owner=owner, repo=repo)


@registry.register(
    name="get_repository_status",
    description="Fetch overall status, latest commit, and health summary of a GitHub repository.",
    parameters={
        "owner": {"type": "string", "description": "Repository owner / organization", "default": DEFAULT_OWNER},
        "repo": {"type": "string", "description": "Repository name", "default": DEFAULT_REPO},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_repository_status(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> Dict[str, Any]:
    return github_client.get_status(owner=owner, repo=repo)


@registry.register(
    name="get_recent_commits",
    description="Fetch the latest commit log for a remote GitHub repository.",
    parameters={
        "owner": {"type": "string", "description": "Repository owner", "default": DEFAULT_OWNER},
        "repo": {"type": "string", "description": "Repository name", "default": DEFAULT_REPO},
        "limit": {"type": "integer", "description": "Max commits to fetch (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_recent_commits(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO, limit: int = 5) -> List[Dict[str, Any]]:
    return github_client.get_commits(owner=owner, repo=repo, limit=limit)


# Backward-compatible tool aliases
@registry.register(
    name="get_github_repo_info",
    description="Fetch repository details and statistics from GitHub (alias for get_repository).",
    parameters={
        "owner": {"type": "string", "description": "Repository owner / organization", "default": DEFAULT_OWNER},
        "repo": {"type": "string", "description": "Repository name", "default": DEFAULT_REPO},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_github_repo_info(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO) -> Dict[str, Any]:
    return get_repository(owner=owner, repo=repo)


@registry.register(
    name="get_github_recent_commits",
    description="Fetch latest commits for a remote GitHub repository (alias for get_recent_commits).",
    parameters={
        "owner": {"type": "string", "description": "Repository owner", "default": DEFAULT_OWNER},
        "repo": {"type": "string", "description": "Repository name", "default": DEFAULT_REPO},
        "limit": {"type": "integer", "description": "Max commits (default 5)", "default": 5},
    },
    permission_level=PermissionLevel.READ_ONLY,
    is_read_only=True,
    risk_level=RiskLevel.LOW,
    isolation=IsolationMode.PROCESS,
    timeout=10.0,
    requires_confirmation=False,
    reversible=True,
)
def get_github_recent_commits(owner: str = DEFAULT_OWNER, repo: str = DEFAULT_REPO, limit: int = 5) -> List[Dict[str, Any]]:
    return get_recent_commits(owner=owner, repo=repo, limit=limit)
