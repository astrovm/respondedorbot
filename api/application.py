"""Objects that make up the running bot.

Think of ``ApplicationRuntime`` as the bot's toolbox. ``api.index`` builds each
tool once, puts it in this object, and entrypoints use the object instead of
importing dozens of unrelated helpers from ``api.index``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from api.admin.service import AdminService
from api.ai.request_runtime import AIRequestService
from api.billing.service import BillingService
from api.cache.service import CacheService
from api.core.config_runtime import ConfigRuntime
from api.markets.dollar import DollarService
from api.bot.giphy import GiphyService
from api.media.images import ImageService
from api.media.cache import MediaCacheService
from api.media.runtime import MediaService
from api.memory.state import MessageStateService
from api.markets.polymarket import PolymarketService
from api.markets.price import PriceService
from api.providers.service import ProviderService
from api.bot.responses import ResponseService
from api.markets.stocks import StockService
from api.memory.summary import SummaryService
from api.services.bcra import BCRAService
from api.bot.telegram import TelegramGateway


@dataclass(frozen=True)
class ApplicationRuntime:
    """The fully wired application shared by Telegram and background jobs.

    Fields are grouped by responsibility: infrastructure first, then business
    services, then the few top-level handlers that receive external events.
    """

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
