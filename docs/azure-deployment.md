# Azure Deployment Guide

## Secret Flow

Preferred order:
1. Azure Key Vault
2. Managed Identity
3. Local environment variables for development only

Secrets are never logged. The secret retrieval service resolves values once and caches them in memory for the process lifetime.

## Required Azure Settings

Set these in App Service, Container Apps, or your host environment:

- `AZURE_KEY_VAULT_URL`
- `AZURE_STORAGE_ACCOUNT_URL`
- `AZURE_BLOB_CONTAINER`
- `AZURE_MONITOR_CONNECTION_STRING` if Azure Monitor export is enabled
- `AZURE_MANAGED_IDENTITY_CLIENT_ID` for user-assigned managed identity, if used

## RBAC Recommendations

Use least privilege and prefer role assignments at the narrowest scope possible.

### Key Vault

Grant the managed identity:
- `Key Vault Secrets User` on the vault or secret scope

If secrets are versioned or scoped narrowly, use the most specific scope available.

### Azure Blob Storage

Grant the managed identity:
- `Storage Blob Data Contributor` on the container or storage account

Do not use shared keys in the application. Container creation defaults to private access.

### Azure Monitor / Log Analytics

When using Azure Monitor OpenTelemetry export:
- Prefer a workspace-based Application Insights resource.
- Store the connection string in Key Vault or App Settings.
- If you later switch to the Logs Ingestion API, grant `Monitoring Data Sender` to the managed identity on the DCR scope.

## Logging Rules

- Log JSON only.
- Never log secret values, prompt text, or document contents.
- Keep request IDs and client IP metadata, but do not log authorization headers.

## Storage Rules

- Keep Blob containers private.
- Do not enable public access.
- Upload only derived artifacts or non-sensitive files.
