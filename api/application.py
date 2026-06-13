from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from api.admin_service import AdminService
from api.ai_request_runtime import AIRequestService
from api.billing_service import BillingService
from api.cache_service import CacheService
from api.config_runtime import ConfigRuntime
from api.dollar_runtime import DollarService
from api.giphy_commands import GiphyService
from api.image_processing import ImageService
from api.media_cache import MediaCacheService
from api.media_runtime import MediaService
from api.message_state import MessageStateService
from api.polymarket_commands import PolymarketService
from api.price_service import PriceService
from api.provider_service import ProviderService
from api.response_runtime import ResponseService
from api.stock_commands import StockService
from api.summary_runtime import SummaryService
from api.services.bcra import BCRAService
from api.telegram_gateway import TelegramGateway


@dataclass(frozen=True)
class ApplicationRuntime:
    config: ConfigRuntime
    telegram: TelegramGateway
    admin: AdminService
    cache: CacheService
    stocks: StockService
    giphy: GiphyService
    polymarket: PolymarketService
    dollar: DollarService
    bcra: BCRAService
    prices: PriceService
    providers: ProviderService
    ai: AIRequestService
    state: MessageStateService
    summary: SummaryService
    responses: ResponseService
    billing: BillingService
    media_cache: MediaCacheService
    media: MediaService
    images: ImageService
    handle_message: Callable[[dict[str, Any]], str]
    handle_callback_query: Callable[[dict[str, Any]], None]
    estimate_ai_base_reserve_credits: Callable[..., Any]


__all__ = ["ApplicationRuntime"]
