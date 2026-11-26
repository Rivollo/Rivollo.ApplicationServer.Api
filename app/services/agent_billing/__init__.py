"""3D Agent (AI Assistant) billing — sql/create_agent_billing.sql.

pricing_service   LLM token usage -> USD cost -> AI credits
hold_service      open / extend a running message's credit hold
settle_service    charge the actual credits, release the rest
usage_service     a seller's usage history (credits only)

Routes: app/api/routes/agent_billing.py
"""
