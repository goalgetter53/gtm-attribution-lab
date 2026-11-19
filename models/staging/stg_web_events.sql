-- materialized: view
-- Pageviews from the product-analytics SDK. Channel is derived from UTM first, then
-- referrer; anything without either is 'direct' (which hides missing-UTM touches).
select
    event_id,
    anonymous_id,
    cast(event_ts as timestamp) as event_ts,
    page,
    lower(utm_source) as utm_source,
    lower(utm_medium) as utm_medium,
    referrer,
    case
        when referrer like '%gtmlab-vendor.com%' then 'internal'
        when lower(utm_medium) in ('cpc', 'ppc', 'paid_search') then 'paid'
        when lower(utm_source) = 'linkedin' or referrer like '%linkedin.com%' then 'linkedin'
        when lower(utm_medium) = 'email' then 'outbound_email'
        when referrer like '%google.%' or referrer like '%bing.%' then 'organic'
        else 'direct'
    end as channel
from {{ source('raw', 'web_events') }}
