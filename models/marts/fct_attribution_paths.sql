-- Touches inside the attribution window that precede each account's conversion event.
-- Grain: one row per (account, touchpoint).
select
    o.account_id,
    t.touchpoint_id,
    t.contact_id,
    t.channel,
    t.touch_ts,
    o.converted,
    o.conversion_at,
    o.mql_at,
    o.opportunity_at,
    t.contact_id = o.focus_contact_id as is_focus_contact,
    (epoch(o.conversion_at) - epoch(t.touch_ts)) / 86400.0 as days_before_conversion,
    row_number() over (
        partition by o.account_id order by t.touch_ts, t.touchpoint_id
    ) as touch_seq
from {{ ref('fct_account_outcomes') }} as o
inner join {{ ref('fct_touchpoints') }} as t on t.account_id = o.account_id
where t.touch_ts <= o.conversion_at
    and t.touch_ts >= o.conversion_at - interval '{{ var('attribution_window_days') }} days'
