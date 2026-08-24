# Image content moderation — how it works and how to tune it

Every user-supplied image is screened by `app/services/image_moderation_service.py`
**before** it is written to blob storage. A rejected image returns HTTP 422
(`CONTENT_POLICY_VIOLATION`) and nothing is stored. GLB/USDZ files are skipped.

Screened routes: `/uploads/content`, `PUT /products/{id}/original-image`,
`/createProductFromGlb` (thumbnail), product background image, public
`/remove-background`.

## Provider

`IMAGE_MODERATION_PROVIDER=azure` (default) uses **Azure AI Content Safety**
(`image:analyze`), which scores each image per category so the operator decides
what kind of images are allowed. If the Azure endpoint/key are not configured,
the service falls back to fal.ai's `imageutils/nsfw` (a single NSFW probability,
no categories, threshold `IMAGE_MODERATION_NSFW_THRESHOLD`).

Azure resource: run `scripts/setup_content_safety.sh <rg> [app-service]` after
`az login`, or Portal → Create a resource → "Content Safety" → copy
*Keys and Endpoint* into:

```
AZURE_CONTENT_SAFETY_ENDPOINT=https://<name>.cognitiveservices.azure.com
AZURE_CONTENT_SAFETY_KEY=<key1>
```

Pricing: F0 = 5,000 images/month free (then the API stops answering — with
`IMAGE_MODERATION_FAIL_OPEN=true` uploads pass unscreened); S0 = pay-as-you-go,
~$0.75 per 1,000 images.

## Severity model

Azure returns a severity per category: **0** safe, **2** low, **4** medium,
**6** high. It classifies photorealistic, illustrated, anime, and AI-generated
imagery alike — animated porn is caught by the Sexual category just like photos.

| Severity | Sexual                                   | Violence                                            |
|----------|------------------------------------------|-----------------------------------------------------|
| 0        | ordinary photo                           | ordinary photo                                      |
| 2        | suggestive / racy                        | weapons shown **as objects** (knife/axe/rifle listing) |
| 4        | nudity, partially explicit               | weapons in use, injuries, blood                     |
| 6        | explicit acts (incl. anime/AI-generated) | graphic gore, death                                 |

## Block rules

`AZURE_CONTENT_SAFETY_BLOCK_RULES` lists the categories that are **not**
allowed and the severity at which they start being rejected. An image is
rejected if ANY listed category scores at or above its threshold; a category
left out is allowed entirely.

Default (recommended):

```
AZURE_CONTENT_SAFETY_BLOCK_RULES=Sexual:2,Violence:4,Hate:4,SelfHarm:4
```

* blocks anything even mildly sexual — including drawn/anime content;
* **allows weapon products** (a knife photographed as a product scores
  Violence 0–2) while blocking gore/injury imagery from severity 4 up;
* blocks clear hate and self-harm imagery.

Known trade-off: sellers listing lingerie/swimwear on models can occasionally
trip `Sexual:2`. If that causes false rejections, relax only that category to
`Sexual:4` (still blocks all nudity and porn). Test borderline images in
[Content Safety Studio](https://contentsafety.cognitive.azure.com) with the
same resource before changing production rules.

## Operational settings

| Env var | Default | Meaning |
|---|---|---|
| `IMAGE_MODERATION_ENABLED` | `true` | Kill switch for all screening |
| `IMAGE_MODERATION_PROVIDER` | `azure` | `azure` or `fal` |
| `IMAGE_MODERATION_FAIL_OPEN` | `true` | Allow uploads when the classifier is unreachable (logged) |
| `IMAGE_MODERATION_TIMEOUT_SECONDS` | `20` | Per-call HTTP budget |
| `AZURE_CONTENT_SAFETY_BLOCK_RULES` | `Sexual:2,Violence:4,Hate:4,SelfHarm:4` | Category:minSeverity list |
| `IMAGE_MODERATION_NSFW_THRESHOLD` | `0.7` | fal fallback only |

Images over Azure's 4 MB request cap are downscaled in memory (Pillow) for the
check only; the original bytes are what gets stored on acceptance.

## Portal behaviour on rejection

`Rivollo.Web.Portal/lib/content-policy.ts`: a rejected image shows a
long-lived "Image not allowed" toast AND locks image uploading for the rest of
the browser session (sessionStorage). Further upload attempts short-circuit
client-side with an "uploads disabled for this session" toast. The lock clears
when the browser/tab session ends; the API screens every request regardless,
so the lock is UX, not security.
