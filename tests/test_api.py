"""Tests for the FastAPI Server."""

import pytest
from fastapi.testclient import TestClient
from backend.api.server import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the health endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "games_active" in data


class TestGameEndpoints:
    """Tests for game-related endpoints."""
    
    def test_create_game(self):
        """Test game creation."""
        response = client.post("/api/v1/games", json={
            "mode": "ai_vs_ai",
            "team_size": 3,
            "time_limit": 300
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "game_id" in data
        assert data["status"] == "created"
        assert "websocket_url" in data
        
        # Cleanup
        game_id = data["game_id"]
        client.delete(f"/api/v1/games/{game_id}")
    
    def test_get_game(self):
        """Test getting game state."""
        # Create a game first
        create_response = client.post("/api/v1/games", json={
            "mode": "ai_vs_ai",
            "team_size": 2
        })
        game_id = create_response.json()["game_id"]
        
        # Get game state
        response = client.get(f"/api/v1/games/{game_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["game_id"] == game_id
        assert "status" in data
        
        # Cleanup
        client.delete(f"/api/v1/games/{game_id}")
    
    def test_get_nonexistent_game(self):
        """Test getting a game that doesn't exist."""
        response = client.get("/api/v1/games/nonexistent_id")
        
        assert response.status_code == 404
    
    def test_control_game(self):
        """Test game control."""
        # Create a game
        create_response = client.post("/api/v1/games", json={"mode": "ai_vs_ai"})
        game_id = create_response.json()["game_id"]
        
        # Test pause
        response = client.post(f"/api/v1/games/{game_id}/control", json={"action": "pause"})
        assert response.status_code == 200
        assert response.json()["game_status"] == "paused"
        
        # Test resume
        response = client.post(f"/api/v1/games/{game_id}/control", json={"action": "resume"})
        assert response.status_code == 200
        assert response.json()["game_status"] == "running"
        
        # Cleanup
        client.post(f"/api/v1/games/{game_id}/control", json={"action": "stop"})
        client.delete(f"/api/v1/games/{game_id}")
    
    def test_delete_game(self):
        """Test game deletion."""
        # Create a game
        create_response = client.post("/api/v1/games", json={"mode": "ai_vs_ai"})
        game_id = create_response.json()["game_id"]
        
        # Delete it
        response = client.delete(f"/api/v1/games/{game_id}")
        
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/games/{game_id}")
        assert get_response.status_code == 404


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_create_game_defaults(self):
        """Test game creation with defaults."""
        response = client.post("/api/v1/games", json={})
        
        assert response.status_code == 200
        
        # Cleanup
        game_id = response.json()["game_id"]
        client.delete(f"/api/v1/games/{game_id}")
    
    def test_invalid_control_action(self):
        """Test invalid control action."""
        # Create a game
        create_response = client.post("/api/v1/games", json={})
        game_id = create_response.json()["game_id"]
        
        # Send invalid action (should still work, just won't do anything specific)
        response = client.post(f"/api/v1/games/{game_id}/control", json={"action": "invalid"})
        assert response.status_code == 200
        
        # Cleanup
        client.delete(f"/api/v1/games/{game_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
