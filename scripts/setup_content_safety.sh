#!/usr/bin/env bash
# Provision Azure AI Content Safety and wire it into the API's App Service.
#
# Usage:
#   az login                                   # interactive, once
#   ./scripts/setup_content_safety.sh <resource-group> [app-service-name] [region] [sku]
#
#   <resource-group>    resource group for the new Content Safety resource
#   [app-service-name]  optional — if given, the settings are pushed to this
#                       App Service; otherwise they are only printed
#   [region]            default: centralindia
#   [sku]               F0 (free, 5k images/mo, hard-stops) or S0 (pay-as-you-go).
#                       Default: F0
#
# Idempotent: re-running reuses the existing resource and re-applies settings.
set -euo pipefail

RG="${1:?resource group required}"
APP="${2:-}"
REGION="${3:-centralindia}"
SKU="${4:-F0}"
NAME="rivollo-content-safety"

echo "==> Creating/ensuring Content Safety resource '$NAME' ($SKU) in $RG/$REGION"
az cognitiveservices account create \
  --name "$NAME" --resource-group "$RG" \
  --kind ContentSafety --sku "$SKU" --location "$REGION" \
  --yes --output none

ENDPOINT=$(az cognitiveservices account show -n "$NAME" -g "$RG" --query properties.endpoint -o tsv)
KEY=$(az cognitiveservices account keys list -n "$NAME" -g "$RG" --query key1 -o tsv)

echo
echo "==> Environment settings for the API:"
cat <<SETTINGS
AZURE_CONTENT_SAFETY_ENDPOINT=${ENDPOINT%/}
AZURE_CONTENT_SAFETY_KEY=$KEY
# defaults shown for completeness — only set to override:
# IMAGE_MODERATION_ENABLED=true
# IMAGE_MODERATION_PROVIDER=azure
# AZURE_CONTENT_SAFETY_BLOCK_RULES=Sexual:2,Violence:4,Hate:4,SelfHarm:4
SETTINGS

if [ -n "$APP" ]; then
  echo "==> Applying settings to App Service '$APP'"
  az webapp config appsettings set -g "$RG" -n "$APP" --output none --settings \
    AZURE_CONTENT_SAFETY_ENDPOINT="${ENDPOINT%/}" \
    AZURE_CONTENT_SAFETY_KEY="$KEY"
  echo "==> Done. Restart the app to pick them up: az webapp restart -g $RG -n $APP"
else
  echo "==> No App Service name given — paste the settings above into"
  echo "    Azure Portal -> App Service -> Configuration, then restart the app."
fi

echo
echo "==> Smoke test:"
echo "curl -s -X POST '\${AZURE_CONTENT_SAFETY_ENDPOINT}/contentsafety/image:analyze?api-version=2024-09-01' \\"
echo "  -H 'Ocp-Apim-Subscription-Key: <key>' -H 'Content-Type: application/json' \\"
echo "  -d '{\"image\":{\"content\":\"'\$(base64 -i test.jpg)'\"}}'"
