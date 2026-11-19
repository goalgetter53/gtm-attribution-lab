-- How much of the raw signal survives identity resolution (one row).
select
    (select count(*) from {{ ref('int_sessions') }}) as web_sessions,
    (
        select count(*)
        from {{ ref('int_sessions') }} as s
        inner join {{ ref('int_identity_map') }} as m on m.anonymous_id = s.anonymous_id
    ) as identified_sessions,
    (select count(*) from {{ ref('int_sessions') }} where landing_channel = 'direct')
        as direct_sessions,
    (select count(*) from {{ ref('int_contact_accounts') }}) as contacts,
    (select count(*) from {{ ref('int_contact_accounts') }} where resolution_method = 'crm')
        as contacts_crm_linked,
    (
        select count(*) from {{ ref('int_contact_accounts') }}
        where resolution_method = 'email_domain'
    ) as contacts_domain_matched,
    (
        select count(*) from {{ ref('int_contact_accounts') }}
        where resolution_method = 'unresolved'
    ) as contacts_unresolved,
    (select count(*) from {{ ref('fct_touchpoints') }}) as resolved_touchpoints
