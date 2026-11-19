select
    t.channel,
    count(*) as touchpoints,
    count(distinct t.account_id) as accounts_touched,
    count(*) filter (where p.converted) as touches_in_won_paths
from {{ ref('fct_touchpoints') }} as t
left join {{ ref('fct_attribution_paths') }} as p on p.touchpoint_id = t.touchpoint_id
group by t.channel
order by touchpoints desc
